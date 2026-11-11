"""Documentation drift checks: links, anchors, module commands and stated counts.

Offline and standard-library only; runs in the normal suite so CI catches stale docs.
"""

import importlib.util
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = sorted([*REPO.glob("*.md"), *REPO.glob("docs/*.md"), *REPO.glob("lessons/*.md"),
               *REPO.glob("datasets/*.md"), *REPO.glob("datasets/hotpotqa/*.md"),
               *REPO.glob("examples/**/*.md")])
# Modules that learners create themselves in the capstone lesson.
LEARNER_MODULES = {"tests.test_my_order_app"}

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
COMMAND = re.compile(r"(?:python3?|uv run python) (?:-B )?-m ([A-Za-z_][\w.]*)((?: tests\.[\w.]+)*)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def prose_lines(text):
    """Yield (line number, line, in_code) so headings inside code fences are ignored."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            yield number, line, True
            continue
        yield number, line, fence is not None


def slug(heading):
    """GitHub-style heading anchor."""
    text = re.sub(r"`([^`]*)`", r"\1", heading.strip())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path, cache={}):
    if path not in cache:
        found, seen = set(), {}
        for _, line, in_code in prose_lines(path.read_text(encoding="utf-8")):
            match = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if match and not in_code:
                base = slug(match.group(2))
                count = seen.get(base, 0)
                seen[base] = count + 1
                found.add(base if count == 0 else f"{base}-{count}")
        cache[path] = found
    return cache[path]


def commands(text):
    for number, line, _ in prose_lines(text):
        for match in COMMAND.finditer(line):
            name = match.group(1)
            # Placeholders such as evaluation.hotpotqa.<module> end the match with a dot.
            if not name.endswith(".") and (REPO / name.split(".")[0]).is_dir():
                yield number, name
            for test in match.group(2).split():
                yield number, test


class DocumentationLinkTests(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self):
        problems = []
        for doc in DOCS:
            for number, line, in_code in prose_lines(doc.read_text(encoding="utf-8")):
                if in_code:
                    continue
                for target in LINK.findall(line):
                    if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                        continue
                    path_part, _, anchor = target.partition("#")
                    path = (doc.parent / path_part).resolve() if path_part else doc
                    where = f"{doc.relative_to(REPO)}:{number}: {target}"
                    if not path.exists():
                        problems.append(f"{where} (missing file)")
                    elif anchor and path.suffix == ".md" and anchor not in anchors(path):
                        problems.append(f"{where} (missing anchor)")
        self.assertEqual(problems, [], "\n".join(problems))


class DocumentationCommandTests(unittest.TestCase):
    def test_module_commands_name_runnable_repository_modules(self):
        problems = []
        for doc in DOCS:
            for number, name in commands(doc.read_text(encoding="utf-8")):
                if name in LEARNER_MODULES:
                    continue
                try:
                    spec = importlib.util.find_spec(name)
                except (ImportError, ValueError):
                    spec = None
                where = f"{doc.relative_to(REPO)}:{number}: {name}"
                if spec is None:
                    problems.append(f"{where} (not found)")
                    continue
                origin = Path(spec.origin) if spec.origin else None
                if origin is None or REPO not in origin.parents:
                    continue  # standard library, e.g. unittest
                if name.split(".")[0] == "tests":
                    continue  # test modules are loaded by unittest, not run directly
                if spec.submodule_search_locations:
                    if not (origin.parent / "__main__.py").exists():
                        problems.append(f"{where} (package without __main__.py)")
                elif '__name__ == "__main__"' not in origin.read_text(encoding="utf-8"):
                    problems.append(f"{where} (no __main__ guard)")
        self.assertEqual(problems, [], "\n".join(problems))


class DocumentationCountTests(unittest.TestCase):
    def read(self, name):
        return " ".join((REPO / name).read_text(encoding="utf-8").split())

    def test_stated_dataset_sizes_match_the_data(self):
        cases = json.loads(self.read("evaluation/support/cases.json"))
        scenarios = json.loads(self.read("evaluation/support/scenarios.json"))
        questions = json.loads(self.read("datasets/hotpotqa/questions.json"))
        cases = cases.get("cases", cases) if isinstance(cases, dict) else cases
        scenarios = scenarios.get("scenarios", scenarios) if isinstance(scenarios, dict) else scenarios
        questions = questions.get("questions", questions) if isinstance(questions, dict) else questions
        self.assertEqual(len(cases), 17)
        self.assertEqual(len(scenarios), 5)
        self.assertEqual(len(questions), 10)
        readme = self.read("README.md")
        for phrase in ("seventeen synthetic", "five multi-turn", "ten development questions"):
            self.assertTrue(phrase in readme, f"README.md no longer says {phrase!r}; update it or this test")
        self.assertTrue("17" in self.read("docs/SUPPORT_EVALUATION.md"))

    def test_python_version_claims_match_pyproject(self):
        required = re.search(r'requires-python = ">=([\d.]+)"', self.read("pyproject.toml")).group(1)
        for doc in DOCS:
            for claim in re.findall(r"Python (\d\.\d+)\+", doc.read_text(encoding="utf-8")):
                self.assertEqual(claim, required, f"{doc.relative_to(REPO)} claims Python {claim}+")


class SlugTests(unittest.TestCase):
    def test_github_style_slugs(self):
        self.assertEqual(slug("Phase 6 checkpoint — review and measurements"),
                         "phase-6-checkpoint--review-and-measurements")
        self.assertEqual(slug("PostgreSQL sessions and journals (optional)"),
                         "postgresql-sessions-and-journals-optional")
        self.assertEqual(slug("1. Purpose and scope"), "1-purpose-and-scope")
        self.assertEqual(slug("`ToolConfig` groups"), "toolconfig-groups")


if __name__ == "__main__":
    unittest.main()
