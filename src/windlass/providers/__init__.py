"""Built-in provider registrations.

Importing this module registers every official Windlass component **by dotted
path**, without importing any of them. That is what keeps ``import windlass``
fast and dependency-free: asking for ``vectordb('pinecone')`` imports the
Pinecone adapter at that moment, and never before.

Each entry records the extras group it needs, so a missing dependency produces
an actionable install command rather than a stack trace.

To see what is available at runtime::

    from windlass import Windlass
    Windlass.list("chunker")

Third-party components register the same way, either with the
:data:`~windlass.core.registry.register` decorators or through the
``windlass.<kind>`` entry-point group.
"""

from __future__ import annotations

from windlass.core.registry import REGISTRY, Registry

__all__ = ["BUILTINS", "register_builtins"]

#: Every built-in component: ``(kind, name, path, extra, aliases, description)``.
#:
#: ``extra`` is the ``windlass`` extras group required; ``None`` means the
#: component works on a bare install with no optional dependencies.
BUILTINS: tuple[tuple[str, str, str, str | None, tuple[str, ...], str], ...] = (
    # -- LLMs ------------------------------------------------------------
    (
        "llm",
        "fake",
        "windlass.providers.llm.fake:FakeLLM",
        None,
        ("mock", "test"),
        "Deterministic in-process model for tests and demos.",
    ),
    (
        "llm",
        "echo",
        "windlass.providers.llm.fake:EchoLLM",
        None,
        (),
        "Returns the last user message verbatim.",
    ),
    (
        "llm",
        "openai",
        "windlass.providers.llm.openai:OpenAILLM",
        "openai",
        ("gpt", "azure-openai"),
        "OpenAI and OpenAI-compatible chat models.",
    ),
    (
        "llm",
        "anthropic",
        "windlass.providers.llm.anthropic:AnthropicLLM",
        "anthropic",
        ("claude",),
        "Anthropic Claude models via the Messages API.",
    ),
    (
        "llm",
        "gemini",
        "windlass.providers.llm.gemini:GeminiLLM",
        "gemini",
        ("google", "google-genai"),
        "Google Gemini models.",
    ),
    (
        "llm",
        "groq",
        "windlass.providers.llm.groq:GroqLLM",
        "groq",
        (),
        "Groq-hosted open models at very high throughput.",
    ),
    (
        "llm",
        "ollama",
        "windlass.providers.llm.ollama:OllamaLLM",
        None,
        ("local",),
        "Locally hosted models via the Ollama daemon.",
    ),
    # -- Embeddings -------------------------------------------------------
    (
        "embedding",
        "hash",
        "windlass.providers.embeddings.hash:HashEmbedder",
        None,
        ("hashing", "fake", "test"),
        "Deterministic hashed n-gram embeddings.",
    ),
    (
        "embedding",
        "huggingface",
        "windlass.providers.embeddings.huggingface:HuggingFaceEmbedder",
        "embeddings",
        ("hf", "sentence-transformers", "st"),
        "Local sentence-transformers models.",
    ),
    (
        "embedding",
        "openai",
        "windlass.providers.embeddings.openai:OpenAIEmbedder",
        "openai",
        (),
        "OpenAI and OpenAI-compatible embedding models.",
    ),
    (
        "embedding",
        "hf_inference",
        "windlass.providers.embeddings.hf_inference:HuggingFaceInferenceEmbedder",
        None,
        ("huggingface-api", "hf-api", "hf"),
        "HuggingFace Inference API embeddings (hosted, no local model).",
    ),
    # -- Vector stores ----------------------------------------------------
    (
        "vectordb",
        "memory",
        "windlass.providers.vectordb.memory:InMemoryVectorStore",
        None,
        ("inmemory", "in-memory", "local"),
        "Exact in-process search with JSON persistence.",
    ),
    (
        "vectordb",
        "faiss",
        "windlass.providers.vectordb.faiss:FaissVectorStore",
        "faiss",
        (),
        "Local approximate search over millions of vectors.",
    ),
    (
        "vectordb",
        "chroma",
        "windlass.providers.vectordb.chroma:ChromaVectorStore",
        "chroma",
        ("chromadb",),
        "ChromaDB, embedded or client/server.",
    ),
    (
        "vectordb",
        "pinecone",
        "windlass.providers.vectordb.pinecone:PineconeVectorStore",
        "pinecone",
        (),
        "Managed, distributed vector search.",
    ),
    # -- Loaders ----------------------------------------------------------
    (
        "loader",
        "text",
        "windlass.providers.loaders.text:TextLoader",
        None,
        ("txt", "plain"),
        "Plain-text files.",
    ),
    (
        "loader",
        "markdown",
        "windlass.providers.loaders.text:MarkdownLoader",
        None,
        ("md",),
        "Markdown with front-matter and title extraction.",
    ),
    (
        "loader",
        "json",
        "windlass.providers.loaders.text:JSONLoader",
        None,
        ("jsonl", "ndjson"),
        "JSON and JSON Lines, one document per record.",
    ),
    (
        "loader",
        "csv",
        "windlass.providers.loaders.text:CSVLoader",
        None,
        ("tsv",),
        "CSV and TSV, one document per row or per file.",
    ),
    (
        "loader",
        "pdf",
        "windlass.providers.loaders.office:PDFLoader",
        "loaders",
        (),
        "PDF documents, one document per page.",
    ),
    (
        "loader",
        "docx",
        "windlass.providers.loaders.office:DocxLoader",
        "loaders",
        ("word", "doc"),
        "Microsoft Word documents.",
    ),
    (
        "loader",
        "pptx",
        "windlass.providers.loaders.office:PptxLoader",
        "loaders",
        ("powerpoint", "ppt"),
        "PowerPoint decks, one document per slide.",
    ),
    (
        "loader",
        "xlsx",
        "windlass.providers.loaders.office:XlsxLoader",
        "loaders",
        ("excel", "xls", "spreadsheet"),
        "Excel workbooks, one document per sheet.",
    ),
    (
        "loader",
        "html",
        "windlass.providers.loaders.web:HTMLLoader",
        None,
        ("htm",),
        "HTML files with boilerplate stripped.",
    ),
    (
        "loader",
        "web",
        "windlass.providers.loaders.web:WebLoader",
        None,
        ("url", "http"),
        "Fetches and extracts text from web pages.",
    ),
    (
        "loader",
        "youtube",
        "windlass.providers.loaders.web:YouTubeLoader",
        "loaders",
        ("yt",),
        "YouTube video transcripts.",
    ),
    (
        "loader",
        "image",
        "windlass.providers.loaders.media:ImageLoader",
        "ocr",
        ("ocr", "png", "jpg"),
        "Images transcribed with Tesseract OCR.",
    ),
    (
        "loader",
        "audio",
        "windlass.providers.loaders.media:AudioLoader",
        "audio",
        ("mp3", "wav", "speech"),
        "Audio transcribed with faster-whisper.",
    ),
    # -- Preprocessors ----------------------------------------------------
    (
        "preprocessor",
        "clean",
        "windlass.providers.preprocessors.clean:CleanPreprocessor",
        None,
        ("cleaner", "normalize"),
        "Normalises whitespace and Unicode, strips boilerplate.",
    ),
    (
        "preprocessor",
        "language",
        "windlass.providers.preprocessors.clean:LanguagePreprocessor",
        None,
        ("lang", "language-filter"),
        "Detects and optionally filters by language.",
    ),
    (
        "preprocessor",
        "metadata",
        "windlass.providers.preprocessors.clean:MetadataPreprocessor",
        None,
        ("enrich", "stats"),
        "Adds computed statistics and keywords.",
    ),
    (
        "preprocessor",
        "pii",
        "windlass.providers.preprocessors.privacy:PIIPreprocessor",
        None,
        ("privacy", "redact"),
        "Detects and redacts personal data.",
    ),
    (
        "preprocessor",
        "dedup",
        "windlass.providers.preprocessors.dedup:DeduplicatePreprocessor",
        None,
        ("deduplicate", "unique"),
        "Removes exact and near-duplicate documents.",
    ),
    (
        "preprocessor",
        "tables",
        "windlass.providers.preprocessors.dedup:TableExtractPreprocessor",
        None,
        ("table-extract",),
        "Splits Markdown tables into their own documents.",
    ),
    (
        "preprocessor",
        "ocr",
        "windlass.providers.preprocessors.enrich:OCRPreprocessor",
        "ocr",
        (),
        "Re-reads scanned pages with OCR when text extraction was empty.",
    ),
    (
        "preprocessor",
        "llm_metadata",
        "windlass.providers.preprocessors.enrich:LLMMetadataPreprocessor",
        None,
        ("extract", "auto-metadata"),
        "Extracts title, summary, topics and entities.",
    ),
    # -- Chunkers ---------------------------------------------------------
    (
        "chunker",
        "recursive",
        "windlass.providers.chunkers.recursive:RecursiveChunker",
        None,
        ("default", "text"),
        "Splits on the largest natural boundary that fits.",
    ),
    (
        "chunker",
        "token",
        "windlass.providers.chunkers.recursive:TokenChunker",
        None,
        ("tokens",),
        "Recursive splitting measured in model tokens.",
    ),
    (
        "chunker",
        "semantic",
        "windlass.providers.chunkers.semantic:SemanticChunker",
        None,
        ("meaning",),
        "Splits at topic shifts detected from sentence embeddings.",
    ),
    (
        "chunker",
        "markdown",
        "windlass.providers.chunkers.structural:MarkdownChunker",
        None,
        ("md",),
        "Splits Markdown on headings, prefixing the heading path.",
    ),
    (
        "chunker",
        "code",
        "windlass.providers.chunkers.structural:CodeChunker",
        None,
        ("source",),
        "Splits source code at function and class boundaries.",
    ),
    (
        "chunker",
        "parent_child",
        "windlass.providers.chunkers.hierarchical:ParentChildChunker",
        None,
        ("parent-child", "small-to-big", "hierarchical"),
        "Indexes small chunks that expand to their parent at query time.",
    ),
    # -- Retrievers -------------------------------------------------------
    (
        "retriever",
        "bm25",
        "windlass.providers.retrievers.bm25:BM25Retriever",
        None,
        ("keyword", "lexical", "sparse"),
        "Okapi BM25 keyword search.",
    ),
    (
        "retriever",
        "vector",
        "windlass.providers.retrievers.vector:VectorRetriever",
        None,
        ("dense", "semantic", "similarity"),
        "Dense embedding search with optional MMR.",
    ),
    (
        "retriever",
        "hybrid",
        "windlass.providers.retrievers.hybrid:HybridRetriever",
        None,
        ("fusion", "rrf", "dense+sparse"),
        "Fuses several retrievers with RRF.",
    ),
    (
        "retriever",
        "contextual",
        "windlass.providers.retrievers.contextual:ContextualRetriever",
        None,
        ("contextual-retrieval", "hyde"),
        "LLM-enriched chunks, optional query rewriting.",
    ),
    # -- Memory -----------------------------------------------------------
    (
        "memory",
        "buffer",
        "windlass.providers.memory.conversation:BufferMemory",
        None,
        ("conversation", "simple"),
        "Keeps the full transcript per thread.",
    ),
    (
        "memory",
        "window",
        "windlass.providers.memory.conversation:WindowMemory",
        None,
        ("sliding", "recent"),
        "Keeps only the most recent turns.",
    ),
    (
        "memory",
        "summary",
        "windlass.providers.memory.conversation:SummaryMemory",
        None,
        ("summarizing", "compressed"),
        "Summarises older turns with an LLM.",
    ),
    (
        "memory",
        "vector",
        "windlass.providers.memory.longterm:VectorMemory",
        None,
        ("long-term", "longterm", "semantic"),
        "Durable facts recalled by similarity.",
    ),
    (
        "memory",
        "composite",
        "windlass.providers.memory.longterm:CompositeMemory",
        None,
        ("hybrid-memory", "both"),
        "Combines conversation and long-term memory.",
    ),
    # -- Guardrails -------------------------------------------------------
    (
        "guardrail",
        "rules",
        "windlass.providers.guardrails.rules:RuleGuardrail",
        None,
        ("rule", "regex", "basic", "default"),
        "Deterministic PII, injection, secret and keyword checks.",
    ),
    (
        "guardrail",
        "nemo",
        "windlass.providers.guardrails.nemo:NeMoGuardrail",
        "guardrails",
        ("nemoguardrails", "nvidia"),
        "NVIDIA NeMo Guardrails.",
    ),
    # -- Evaluators -------------------------------------------------------
    (
        "evaluator",
        "builtin",
        "windlass.providers.evaluation.builtin:BuiltinEvaluator",
        None,
        ("default", "windlass"),
        "Lexical and LLM-judged RAG metrics.",
    ),
    (
        "evaluator",
        "ragas",
        "windlass.providers.evaluation.external:RagasEvaluator",
        "evaluation",
        (),
        "RAGAS reference-free RAG metrics.",
    ),
    (
        "evaluator",
        "deepeval",
        "windlass.providers.evaluation.external:DeepEvalEvaluator",
        "evaluation",
        (),
        "DeepEval metrics with pass/fail verdicts.",
    ),
    # -- Tracers ----------------------------------------------------------
    (
        "tracer",
        "console",
        "windlass.providers.observability.console:ConsoleTracer",
        None,
        ("stdout", "print", "debug"),
        "Prints an indented trace tree.",
    ),
    (
        "tracer",
        "memory",
        "windlass.providers.observability.console:MemoryTracer",
        None,
        ("collect", "test"),
        "Collects spans in a list for assertions.",
    ),
    (
        "tracer",
        "null",
        "windlass.interfaces.tracer:NullTracer",
        None,
        ("none", "off"),
        "Discards every span.",
    ),
    (
        "tracer",
        "langsmith",
        "windlass.providers.observability.platforms:LangSmithTracer",
        "observability",
        ("langchain",),
        "Exports traces to LangSmith.",
    ),
    (
        "tracer",
        "multi",
        "windlass.providers.observability.multi:MultiTracer",
        None,
        ("fanout", "fan-out", "tee"),
        "Sends every span to several tracing backends at once.",
    ),
    (
        "tracer",
        "langfuse",
        "windlass.providers.observability.platforms:LangfuseTracer",
        "observability",
        (),
        "Exports traces to Langfuse.",
    ),
    # -- Caches -----------------------------------------------------------
    (
        "cache",
        "memory",
        "windlass.core.cache:MemoryCache",
        None,
        ("inmemory", "lru"),
        "Thread-safe in-process cache with TTL and LRU eviction.",
    ),
    (
        "cache",
        "disk",
        "windlass.core.cache:DiskCache",
        "cache",
        ("persistent",),
        "Persistent cache shared across processes.",
    ),
    (
        "cache",
        "null",
        "windlass.core.cache:NullCache",
        None,
        ("none", "off"),
        "Discards everything; used when caching is disabled.",
    ),
    # -- Checkpointers ----------------------------------------------------
    (
        "checkpointer",
        "memory",
        "windlass.agent.checkpoint:MemoryCheckpointer",
        None,
        ("inmemory", "default"),
        "In-process checkpoint store.",
    ),
    (
        "checkpointer",
        "sqlite",
        "windlass.agent.checkpoint:SQLiteCheckpointer",
        None,
        ("file", "durable"),
        "Durable checkpoint store backed by SQLite.",
    ),
    # -- MCP --------------------------------------------------------------
    (
        "mcp",
        "fastmcp",
        "windlass.providers.mcp.fastmcp:FastMCPClient",
        "mcp",
        ("mcp", "server"),
        "Connects to MCP servers over stdio, SSE or HTTP.",
    ),
    (
        "mcp",
        "static",
        "windlass.providers.mcp.fastmcp:StaticMCPClient",
        None,
        ("inprocess", "fake"),
        "In-process MCP client for tests.",
    ),
    (
        "mcp",
        "multi",
        "windlass.providers.mcp.fastmcp:MultiMCPClient",
        None,
        ("aggregate",),
        "Aggregates several MCP servers behind one client.",
    ),
)


def register_builtins(registry: Registry | None = None) -> int:
    """Register every built-in component lazily.

    Called once when :mod:`windlass` is imported. Safe to call again — entries are
    replaced rather than duplicated, which makes it usable to restore the
    registry after a test has torn it down.

    Args:
        registry: Registry to populate. Defaults to the global one.

    Returns:
        How many components were registered.

    Example:
        >>> from windlass.core.registry import Registry
        >>> reg = Registry()
        >>> register_builtins(reg) > 40
        True
        >>> "recursive" in reg.names("chunker")
        True
    """
    target = registry or REGISTRY
    for kind, name, path, extra, aliases, description in BUILTINS:
        target.register_lazy(
            kind,
            name,
            path,
            aliases=aliases,
            description=description,
            extra=extra,
            origin="builtin",
            override=True,
        )
    return len(BUILTINS)
