"""Offline checks for the local storage test runner; no Docker or databases are used."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import run_storage_tests as runner

DEFAULT_ENVIRONMENT = runner.default_environment


def outcome(success=True, skipped=()):
    return SimpleNamespace(wasSuccessful=lambda: success, skipped=[(None, reason) for reason in skipped])


class RunStorageTestsTests(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(runner, "compose", return_value=0),
                   mock.patch.object(runner.check_storage_services, "main", return_value=0),
                   mock.patch.object(runner, "default_environment"),
                   mock.patch.object(runner, "run_tests", return_value=outcome())]
        self.compose, self.check, _, self.tests = (p.start() for p in patches)
        for patch in patches:
            self.addCleanup(patch.stop)

    def test_defaults_respect_existing_values_and_port_overrides(self):
        environ = {"LITTLE_HARNESS_POSTGRES_PORT": "6000", runner.check_storage_services.MONGODB_ENV: "kept"}
        DEFAULT_ENVIRONMENT(environ, env_file="missing.env")
        self.assertEqual(environ[runner.check_storage_services.POSTGRES_ENV],
                         "postgresql://postgres@127.0.0.1:6000/postgres")
        self.assertEqual(environ[runner.check_storage_services.MONGODB_ENV], "kept")

    def test_ports_come_from_env_file_but_shell_wins_and_nothing_else_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("LITTLE_HARNESS_POSTGRES_PORT=6001\nLITTLE_HARNESS_MONGODB_PORT=6002\n"
                                "ANTHROPIC_API_KEY=secret\n")
            environ = {"LITTLE_HARNESS_MONGODB_PORT": "7002"}
            DEFAULT_ENVIRONMENT(environ, env_file=env_file)
        self.assertIn(":6001/", environ[runner.check_storage_services.POSTGRES_ENV])
        self.assertIn(":7002/", environ[runner.check_storage_services.MONGODB_ENV])
        self.assertNotIn("ANTHROPIC_API_KEY", environ)
        self.assertNotIn("LITTLE_HARNESS_POSTGRES_PORT", environ)

    def test_success_starts_databases_and_runs_storage_then_full_suite(self):
        self.assertEqual(runner.main([]), 0)
        self.compose.assert_called_once_with("up", "-d", "--wait")
        self.assertEqual(self.tests.call_args_list, [mock.call(runner.SUITES), mock.call()])

    def test_skips_for_missing_database_or_driver_fail(self):
        for reason in ("set LITTLE_HARNESS_TEST_POSTGRES_DSN and install the postgres extra",
                       "install the mongodb extra"):
            self.tests.return_value = outcome(skipped=[reason])
            with self.subTest(reason=reason), mock.patch("builtins.print"):
                self.assertEqual(runner.main(["--no-start", "--storage-only"]), 1)
        self.compose.assert_not_called()

    def test_by_design_skips_are_allowed(self):
        self.tests.return_value = outcome(skipped=["this store assumes a single writer",
                                                  "the postgres extra is installed"])
        self.assertEqual(runner.main(["--no-start", "--storage-only"]), 0)

    def test_failed_service_check_stops_before_tests_and_down_still_runs(self):
        self.check.return_value = 1
        self.assertEqual(runner.main(["--down"]), 1)
        self.tests.assert_not_called()
        self.assertEqual(self.compose.call_args_list[-1], mock.call("down"))

    def test_storage_only_skips_full_suite(self):
        self.assertEqual(runner.main(["--no-start", "--storage-only"]), 0)
        self.tests.assert_called_once_with(runner.SUITES)


if __name__ == "__main__":
    unittest.main()
