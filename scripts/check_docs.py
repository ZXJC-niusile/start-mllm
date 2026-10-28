"""Check local Markdown paths, explicit fragment IDs and code fences offline."""
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check_file(path, root):
    source = path.read_text(encoding="utf-8")
    errors = []
    fence = None
    prose = []
    for number, line in enumerate(source.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence is None:
            prose.append((number, line))
    if fence:
        errors.append(f"{path}: unclosed code fence")
    count = 0
    for number, line in prose:
        for match in re.finditer(r"\]\(([^)]+)\)", line):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            raw = unquote(url.path)
            destination = (root / "docs" / raw.lstrip("/")) if raw.startswith("/") else path.parent / raw
            if not raw:
                destination = path
            count += 1
            if not destination.exists():
                errors.append(f"{path}:{number}: missing local path {target}")
                continue
            if url.fragment and destination.suffix == ".md":
                content = destination.read_text(encoding="utf-8")
                fragment = unquote(url.fragment)
                # Cross-page links use explicit IDs so Docsify and GitHub agree.
                ids = re.findall(r"\bid=[\"']([^\"']+)[\"']", content)
                if fragment not in ids:
                    errors.append(f"{path}:{number}: missing explicit anchor {target}")
    return count, errors


def main() -> int:
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "*.md"], cwd=ROOT).decode("utf-8").split("\0")
    errors, count = [], 0
    for relative in filter(None, tracked):
        checked, problems = check_file(ROOT / relative, ROOT)
        count += checked
        errors.extend(problems)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Local links checked: {count}; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
