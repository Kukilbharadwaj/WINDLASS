"""The ``windlass`` command-line interface.

A small set of commands that answer the questions people actually have when
something is not working::

    windlass info                 # version, Python, which extras are installed
    windlass list                 # every registered component
    windlass list chunker         # one kind, with descriptions
    windlass config               # effective settings, secrets masked
    windlass doctor               # diagnose a broken install
    windlass ask "..." --docs ./  # a one-shot RAG query
    windlass chat                 # an interactive agent session

``doctor`` is the important one: it reports exactly which optional dependencies
are present, whether credentials are configured, and what the defaults resolve
to — which is usually enough to explain the problem without opening a REPL.
"""

from __future__ import annotations

import argparse
import sys

from windlass._version import __version__

__all__ = ["build_parser", "main"]

#: Extras group to the module that proves it is installed.
_EXTRAS: dict[str, tuple[str, ...]] = {
    "openai": ("openai",),
    "anthropic": ("anthropic",),
    "gemini": ("google.genai",),
    "groq": ("groq",),
    "agent": ("langgraph",),
    "embeddings": ("sentence_transformers", "numpy"),
    "faiss": ("faiss",),
    "chroma": ("chromadb",),
    "pinecone": ("pinecone",),
    "loaders": ("pypdf", "docx", "pptx", "openpyxl", "bs4"),
    "mcp": ("fastmcp",),
    "evaluation": ("ragas", "deepeval"),
    "guardrails": ("nemoguardrails",),
    "observability": ("langsmith", "langfuse"),
    "ocr": ("pytesseract", "PIL"),
    "audio": ("faster_whisper",),
    "cache": ("diskcache",),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="windlass",
        description="Windlass — the modular AI application framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  windlass doctor\n"
            "  windlass list retriever\n"
            '  windlass ask "What is in these docs?" --docs ./documents\n'
            "  windlass chat --llm ollama --model llama3.2\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"windlass {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("info", help="show version and environment information")

    listing = sub.add_parser("list", help="list registered components")
    listing.add_argument("kind", nargs="?", help="component kind, e.g. chunker")
    listing.add_argument("--verbose", "-v", action="store_true", help="show descriptions")

    config = sub.add_parser("config", help="show the effective settings")
    config.add_argument("--json", action="store_true", help="emit JSON")

    sub.add_parser("doctor", help="diagnose the installation")

    ask = sub.add_parser("ask", help="run a one-shot RAG query")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument("--docs", "-d", required=True, help="file or directory to index")
    ask.add_argument("--llm", default=None, help="model provider")
    ask.add_argument("--model", default=None, help="model id")
    ask.add_argument("--embedding", default=None, help="embedding provider")
    ask.add_argument("--chunker", default=None, help="chunking strategy")
    ask.add_argument("--retriever", default=None, help="retrieval strategy")
    ask.add_argument("--top-k", type=int, default=5, help="chunks to retrieve")
    ask.add_argument("--trace", action="store_true", help="print a trace tree")

    chat = sub.add_parser("chat", help="start an interactive agent session")
    chat.add_argument("--llm", default=None, help="model provider")
    chat.add_argument("--model", default=None, help="model id")
    chat.add_argument("--system", default=None, help="system prompt")
    chat.add_argument("--trace", action="store_true", help="print a trace tree")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on success, ``1`` on error, ``130`` on
        interrupt.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "info": _info,
        "list": _list,
        "config": _config,
        "doctor": _doctor,
        "ask": _ask,
        "chat": _chat,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        _out("\nInterrupted.")
        return 130
    except Exception as exc:
        from windlass.core.exceptions import WindlassError

        if isinstance(exc, WindlassError):
            _out(f"\nError: {exc}")
        else:
            _out(f"\nUnexpected {type(exc).__name__}: {exc}")
        return 1


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def _info(args: argparse.Namespace) -> int:
    """Print version and environment information."""
    import platform

    from windlass.core.registry import REGISTRY

    _out(f"windlass {__version__}")
    _out(f"python     {platform.python_version()} ({platform.system()})")
    total = sum(len(REGISTRY.names(k)) for k in REGISTRY.kinds())
    _out(f"components {total} registered across {len(REGISTRY.kinds())} kinds")

    installed = [name for name, modules in _EXTRAS.items() if _has_all(modules)]
    partial = [
        name
        for name, modules in _EXTRAS.items()
        if name not in installed and any(_has(m) for m in modules)
    ]
    _out(f"extras     {', '.join(installed) if installed else '(core only)'}")
    if partial:
        _out(f"partial    {', '.join(partial)}")
    return 0


def _list(args: argparse.Namespace) -> int:
    """List registered components."""
    from windlass.core.registry import REGISTRY

    kinds = [args.kind] if args.kind else REGISTRY.kinds()
    if args.kind and args.kind not in REGISTRY.kinds():
        _out(f"Unknown kind {args.kind!r}. Known kinds: {', '.join(REGISTRY.kinds())}")
        return 1

    for kind in kinds:
        specs = REGISTRY.specs(kind)
        if not specs:
            continue
        _out(f"\n{kind}")
        for spec in specs:
            marker = "" if spec.extra is None or _extra_ok(spec.extra) else "  (not installed)"
            if args.verbose or args.kind:
                aliases = f"  [{', '.join(spec.aliases)}]" if spec.aliases else ""
                _out(f"  {spec.name:<16}{marker}{aliases}")
                if spec.description:
                    _out(f"      {spec.description}")
            else:
                _out(f"  {spec.name}{marker}")
    return 0


def _config(args: argparse.Namespace) -> int:
    """Print the effective settings with secrets masked."""
    from windlass.core.config import settings

    data = settings().masked()
    if args.json:
        import json

        _out(json.dumps(data, indent=2, default=str))
        return 0

    for key in sorted(data):
        value = data[key]
        if isinstance(value, dict):
            _out(f"{key}:")
            for inner in sorted(value):
                _out(f"  {inner:<20} {value[inner]}")
        else:
            _out(f"{key:<24} {value}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Diagnose the installation and report anything that looks wrong."""
    from windlass.core.config import settings
    from windlass.core.registry import REGISTRY

    _out("Windlass doctor\n" + "=" * 40)
    problems: list[str] = []

    _out("\nExtras:")
    for name, modules in sorted(_EXTRAS.items()):
        present = [m for m in modules if _has(m)]
        if len(present) == len(modules):
            status = "ok"
        elif present:
            status = f"partial ({', '.join(present)})"
        else:
            status = "not installed"
        _out(f"  {name:<16} {status}")

    cfg = settings()
    _out("\nCredentials:")
    for label, field in (
        ("OpenAI", "openai_api_key"),
        ("Anthropic", "anthropic_api_key"),
        ("Google", "google_api_key"),
        ("Groq", "groq_api_key"),
        ("HuggingFace", "huggingface_api_key"),
        ("Pinecone", "pinecone_api_key"),
        ("LangSmith", "langsmith_api_key"),
        ("Langfuse", "langfuse_secret_key"),
    ):
        _out(f"  {label:<16} {'configured' if cfg.secret(field) else '-'}")

    _out("\nDefaults:")
    for label, value in (
        ("llm", cfg.default_llm),
        ("embedding", cfg.default_embedding),
        ("vectordb", cfg.default_vectordb),
        ("chunker", cfg.default_chunker),
        ("retriever", cfg.default_retriever),
    ):
        ok = REGISTRY.has(label if label != "embedding" else "embedding", value)
        _out(f"  {label:<16} {value}{'' if ok else '  (NOT REGISTERED)'}")
        if not ok:
            problems.append(f"default {label} {value!r} is not registered")

    _out("\nSmoke test:")
    try:
        from windlass.api import Windlass

        rag = Windlass.rag()
        rag.ingest_text("Windlass is a modular AI application framework.")
        answer = rag.ask("What is Windlass?")
        _out(f"  offline pipeline ok ({len(answer.contexts)} context(s) retrieved)")
    except Exception as exc:
        _out(f"  offline pipeline FAILED: {exc}")
        problems.append(f"offline pipeline failed: {exc}")

    if problems:
        _out(f"\n{len(problems)} problem(s) found:")
        for problem in problems:
            _out(f"  - {problem}")
        return 1
    _out("\nEverything looks healthy.")
    return 0


def _ask(args: argparse.Namespace) -> int:
    """Index a path and answer one question against it."""
    from windlass.api import Windlass

    rag = Windlass.rag()
    if args.llm:
        rag.llm(args.llm, **({"model": args.model} if args.model else {}))
    elif args.model:
        rag.llm(args.model)
    if args.embedding:
        rag.embedding(args.embedding)
    if args.chunker:
        rag.chunker(args.chunker)
    if args.retriever:
        rag.retriever(args.retriever)
    if args.trace:
        rag.observe("console", show_io=True)
    rag.top_k(args.top_k)

    _out(f"Indexing {args.docs} …")
    chunks = rag.ingest(args.docs)
    _out(f"Indexed {chunks} chunk(s).\n")

    answer = rag.ask(args.question)
    _out(answer.answer)
    if answer.sources:
        _out("\nSources:")
        for source in answer.sources:
            _out(f"  - {source}")
    _out(f"\n({answer.usage.total_tokens} tokens, {answer.latency_ms:.0f}ms)")
    return 0


def _chat(args: argparse.Namespace) -> int:
    """Run an interactive agent session."""
    from windlass.api import Windlass

    agent = Windlass.agent().memory("window", window=20)
    if args.llm:
        agent.llm(args.llm, **({"model": args.model} if args.model else {}))
    elif args.model:
        agent.llm(args.model)
    if args.system:
        agent.system(args.system)
    if args.trace:
        agent.observe("console")

    _out("Windlass chat. Type 'exit' to quit.\n")
    thread = "cli"
    while True:
        try:
            message = input("you › ").strip()
        except EOFError:
            _out("")
            return 0
        if not message:
            continue
        if message.lower() in {"exit", "quit", ":q"}:
            return 0

        sys.stdout.write("bot › ")
        sys.stdout.flush()
        try:
            for event in agent.stream(message, thread_id=thread):
                if event.type == "text" and event.delta:
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                elif event.type == "tool_call" and event.tool_call:
                    sys.stdout.write(f"\n  [calling {event.tool_call.name}]\n")
                    sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(f"\n  error: {exc}")
        _out("\n")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _out(text: str) -> None:
    """Write a line to stdout.

    Wrapped so the linter's no-print rule stays useful everywhere else in the
    codebase, where printing really would be wrong.
    """
    sys.stdout.write(f"{text}\n")


def _has(module: str) -> bool:
    """Return whether a module is importable."""
    from windlass.core.lazy import is_available

    return is_available(module)


def _has_all(modules: tuple[str, ...]) -> bool:
    """Return whether every module in a group is importable."""
    return all(_has(m) for m in modules)


def _extra_ok(extra: str) -> bool:
    """Return whether an extras group appears to be installed."""
    modules = _EXTRAS.get(extra)
    return _has_all(modules) if modules else True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
