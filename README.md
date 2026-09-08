# Ask My Docs

A fully local RAG application. It reads documents from `documents`, stores the
search index locally in Chroma, and uses your local Ollama models for embeddings
and answers. Nothing is sent to a cloud service.

Supports PDF, Word (`.docx`), Excel (`.xls`, `.xlsx`, `.xlsm`), CSV, text, and Markdown. It uses LangChain's `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredExcelLoader`, and `TextLoader`.

```powershell
# Install Python dependencies
uv sync

# Install Ollama models (after installing Ollama from https://ollama.com)
ollama pull llama3.1
ollama pull nomic-embed-text

# Build/update the search index from documents/
uv run python main.py ingest

# Ask a single question
uv run python main.py ask "What products are described?"

# Or chat interactively
uv run python main.py chat

# Or start the browser interface, then open http://127.0.0.1:8000
uv run uvicorn web:app --reload
```

Use a different source folder or model if wanted:

```powershell
uv run python main.py --documents-dir C:\my-docs --model llama3.1 ingest
uv run python main.py --embedding-model nomic-embed-text chat
```

Use `ingest --rebuild` to discard and recreate the local index. `.rag_db` is
excluded from Git.

## Browser interface

The web app displays a question form, answer, and the document sources used for
that answer. Select **Index documents** after adding or changing files in
`documents`. It runs locally at `http://127.0.0.1:8000` and uses the same local
Ollama models and Chroma index as the command-line app.
