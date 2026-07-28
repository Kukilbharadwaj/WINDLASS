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

Rehearse against TestPyPI first if you want (needs a separate token from
<https://test.pypi.org/manage/account/token/>):

```powershell
.\venv\Scripts\python.exe -m twine upload --repository testpypi dist/*
```

## 3. Verify

```bash
pip install windlass
python -c "import windlass; print(windlass.__version__)"
windlass doctor
```

## Pre-flight state (already done)

| Check | Result |
|---|---|
| Name `windlass` free on PyPI | yes, verified |
| `twine check` on wheel + sdist | PASSED |
| Wheel installs in a clean venv | yes — 15 dependencies |
| Sdist builds and installs from source | yes |
| `windlass` CLI entry point | works |
| Secrets in artifacts | none — no `.env` included |
| Test suite | 652 passed |
| ruff / mypy | clean |

## After publishing

**A version number on PyPI can never be reused**, even if you delete the release.
If `0.1.0` needs a fix, bump to `0.1.1` in `src/windlass/_version.py`, rebuild
(`python -m build`) and upload again.
