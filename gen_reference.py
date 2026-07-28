"""Generate the API reference pages from the source tree.

Run automatically by ``mkdocs-gen-files`` during a build. Every public module
gets a page containing a single ``:::`` directive, and mkdocstrings renders the
Google-style docstrings that are already in the code — so the reference cannot
drift from the implementation.

Private modules (a leading underscore) are skipped, as are ``__main__`` files.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SOURCE = Path("src")
PACKAGE = "windlass"

navigation = mkdocs_gen_files.Nav()

for path in sorted(SOURCE.rglob("*.py")):
    module_path = path.relative_to(SOURCE).with_suffix("")
    doc_path = path.relative_to(SOURCE).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    if not parts or any(part.startswith("_") and part != "__init__" for part in parts[1:-1]):
        continue

    navigation[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as handle:
        identifier = ".".join(parts)
        handle.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("..", path))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as handle:
    handle.writelines(navigation.build_literate_nav())
