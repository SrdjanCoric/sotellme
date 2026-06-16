"""Stage the canonical README and LICENSE into the package for a build.

The package lives in ``backend/`` while the README and LICENSE are the repository's
front door at the root. ``uv build`` isolates the build to ``backend/``, so a symlink
to ``../README.md`` dangles. This copies the real files in beside ``pyproject.toml``
just before the build; the copies are gitignored so the root stays the single source.
"""

import shutil
import sys
from pathlib import Path

_FILES = ("README.md", "LICENSE")


def stage_package_files(package_dir: Path, repo_root: Path) -> None:
    for name in _FILES:
        source = repo_root / name
        if not source.exists():
            raise FileNotFoundError(f"Cannot stage {name}: {source} does not exist.")
        shutil.copyfile(source, package_dir / name)


def main() -> int:
    package_dir = Path(__file__).resolve().parent.parent
    stage_package_files(package_dir, package_dir.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
