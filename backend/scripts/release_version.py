"""Compute and stamp the build version for a CI publish.

A branch push stages a unique dev version (`<base>.dev<run-number>`); a release tag
publishes the tag's own version. The pure functions are unit-tested; `main` is the
thin CLI the release workflow calls to stamp `pyproject.toml` before building.
"""

import argparse
import re
import sys
from pathlib import Path

_RELEASE_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_VERSION_FIELD = re.compile(r'(?m)^(version\s*=\s*)"[^"]*"')


def compute_version(base: str, ref_type: str, ref_name: str, run_number: int) -> str:
    if ref_type == "tag":
        version = ref_name[1:] if ref_name.startswith("v") else ref_name
        if not _RELEASE_VERSION.match(version):
            raise ValueError(f"Tag {ref_name!r} is not a release version (expected vX.Y.Z).")
        return version
    return f"{base}.dev{run_number}"


def read_base_version(pyproject_text: str) -> str:
    match = _VERSION_FIELD.search(pyproject_text)
    if match is None:
        raise ValueError("No version field found in pyproject.toml.")
    return match.group(0).split('"')[1]


def replace_version(pyproject_text: str, new_version: str) -> str:
    rewritten, count = _VERSION_FIELD.subn(rf'\g<1>"{new_version}"', pyproject_text)
    if count != 1:
        raise ValueError(f"Expected exactly one version field, found {count}.")
    return rewritten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-type", required=True, help="git ref type: 'branch' or 'tag'.")
    parser.add_argument("--ref-name", required=True, help="git ref name, e.g. 'main' or 'v0.2.0'.")
    parser.add_argument("--run-number", required=True, type=int, help="CI run number.")
    parser.add_argument(
        "--pyproject", type=Path, default=Path("pyproject.toml"), help="Path to pyproject.toml."
    )
    args = parser.parse_args(argv)

    text = args.pyproject.read_text()
    base = read_base_version(text)
    version = compute_version(base, args.ref_type, args.ref_name, args.run_number)
    args.pyproject.write_text(replace_version(text, version))
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
