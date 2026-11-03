"""Provider-module isolation checks; protocol/transport coverage also lives in test_mini_harness."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ProviderIsolationTests(unittest.TestCase):
    def test_provider_import_and_stream_parsing_need_no_harness_or_workspace(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="provider-isolation-test-") as temporary:
            root = Path(temporary)
            (root / ".env").write_text("HARNESS_MODEL=should-not-load\n")
            code = (
                "import json, os, sys; from pathlib import Path; from harness import model_clients as c; "
                "assert 'mini_harness' not in sys.modules; "
                "assert 'HARNESS_MODEL' not in os.environ; "
                "assert not Path('workspace').exists(); "
                "event = {'choices': [{'delta': {'content': 'hello'}, 'finish_reason': 'stop'}]}; "
                "lines = [('data: ' + json.dumps(event) + '\\n').encode(), b'\\n', "
                "b'data: [DONE]\\n', b'\\n']; "
                "reply = c.OpenAIChatModel('local')._read_stream(lines); "
                "assert reply['content'] == [{'type': 'text', 'text': 'hello'}]; "
                "assert reply['stop_reason'] == 'end_turn'"
            )
            child = subprocess.run([sys.executable, "-c", code], cwd=root,
                                   env={"PATH": os.defpath, "PYTHONPATH": str(repo)},
                                   capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertFalse((root / "workspace").exists())


if __name__ == "__main__":
    unittest.main()