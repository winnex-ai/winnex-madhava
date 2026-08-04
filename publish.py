#!/usr/bin/env python3
"""
Publish winnex-madhava to PyPI.

Builds the wheel + sdist and uploads them. Production uploads use the PyPI
Trusted Publisher (OpenID Connect) from GitHub Actions; this script is the
convenience path for manual/local uploads (e.g. to TestPyPI) and for anyone
running the build outside CI.

Usage:
  python publish.py                # build + upload to TestPyPI
  python publish.py --prod         # build + upload to PyPI (real)
  python publish.py --no-upload    # build only (wheel + sdist in dist/)
  python publish.py --help         # this message

Requires: pip install build twine. Upload credentials come from
~/.pypirc or TWINE_USERNAME / TWINE_PASSWORD (never from argv).
"""
import argparse
import glob
import shutil
import subprocess
import sys


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def build() -> None:
    for pat in ("dist", "build", "*.egg-info"):
        for p in glob.glob(pat):
            shutil.rmtree(p, ignore_errors=True)
    sh(f"{sys.executable} -m build")  # wheel + sdist


def upload(repo: str) -> None:
    sh(f"{sys.executable} -m twine upload --repository {repo} dist/*")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and optionally upload winnex-madhava to PyPI.",
        epilog=(
            "Uploads require twine credentials via ~/.pypirc or the "
            "TWINE_USERNAME / TWINE_PASSWORD environment variables. "
            "Never pass a token on the command line."
        ),
    )
    ap.add_argument(
        "--prod",
        action="store_true",
        help="upload to the real PyPI (default is TestPyPI)",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="build only — produce dist/ artifacts, do not upload",
    )
    ap.add_argument(
        "--repository",
        default=None,
        help='twine repository name (default: "testpypi", or "pypi" with --prod)',
    )
    args = ap.parse_args()

    repo = args.repository or ("pypi" if args.prod else "testpypi")
    print(f"=== Publicando winnex-madhava para {repo} ===")

    build()

    if args.no_upload:
        print("\nOK. Build complete — dist/ contains the artifacts (not uploaded).")
        return

    upload(repo)

    host = "pypi.org" if repo == "pypi" else "test.pypi.org"
    print(f"\nOK. Publicado em https://{host}/project/winnex-madhava/")
    print("Instale com: pip install winnex-madhava")


if __name__ == "__main__":
    main()
