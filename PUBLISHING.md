# Publishing Windlass to PyPI

Everything is built, validated and verified. Publishing is one command; it needs
a PyPI API token, which only the project owner can mint.

## 1. Get a token

* Create the account / log in: <https://pypi.org/account/register/>
* Mint a token: <https://pypi.org/manage/account/token/>
  * Scope: **"Entire account"** for the very first upload — a project-scoped
    token cannot be created until the project exists on PyPI.
  * After the first release, replace it with a token scoped to `windlass` only.
* The token starts with `pypi-`. Treat it like a password.

## 2. Publish

From the repository root, with the project virtualenv active:

```powershell
# Windows PowerShell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR-TOKEN-HERE"
.\venv\Scripts\python.exe -m twine upload dist/*
```

```bash
# macOS / Linux
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
python -m twine upload dist/*
```

## 2a. Rehearse on TestPyPI first (recommended)

TestPyPI is a throwaway copy of PyPI. Upload there, install from there, confirm
the project page and the install both look right — then do the real upload. It
needs its own account and token from <https://test.pypi.org/manage/account/token/>;
a PyPI token will **not** work on TestPyPI.

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR-TESTPYPI-TOKEN"
.\venv\Scripts\python.exe -m twine upload --repository testpypi dist/*
```

Then install it into a scratch environment. `--extra-index-url` is required
because Windlass's own dependencies (pydantic, httpx, ...) live on real PyPI,
not on TestPyPI:

```powershell
py -m venv C:\Temp\wl-test
C:\Temp\wl-test\Scripts\python.exe -m pip install `
    --index-url https://test.pypi.org/simple/ `
    --extra-index-url https://pypi.org/simple/ `
    windlass
C:\Temp\wl-test\Scripts\windlass.exe doctor
```

Check <https://test.pypi.org/project/windlass/> renders the README correctly and
the sidebar links resolve. Uploading the same 0.1.0 to real PyPI afterwards is
fine — TestPyPI and PyPI are entirely separate registries.

## 3. Verify

```bash
pip install windlass
python -c "import windlass; print(windlass.__version__)"
windlass doctor
```

## Pre-flight state (verified)

| Check | Result |
|---|---|
| Name `windlass` free on PyPI | yes — `/pypi/windlass/json` and `/simple/windlass/` both 404 |
| `twine check` on wheel + sdist | PASSED |
| Wheel installs in a clean venv | yes — 15 packages total, all pure-Python |
| Sdist builds and installs from source | yes |
| `windlass` CLI entry point | works — `windlass doctor` reports healthy |
| Secrets in artifacts | none — no `.env`, no venv, no `.git` |
| Test suite | 652 passed, 21 skipped, ~40s |
| ruff / mypy | clean |
| Project URLs resolve | point at `Kukilbharadwaj/WINDLASS` — **make that repo public before release**, or the PyPI sidebar links 404 |

Rebuild (`python -m build`) after **any** change to `README.md` or
`pyproject.toml` — the README is the PyPI landing page, and it is baked into the
artifacts at build time, not read at upload time.

## After publishing

**A version number on PyPI can never be reused**, even if you delete the release.
If `0.1.0` needs a fix, bump to `0.1.1` in `src/windlass/_version.py`, rebuild
(`python -m build`) and upload again.
