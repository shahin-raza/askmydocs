"""Local RAG app: `uv run python main.py ingest`, `chat`, or `ask "question"`."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
SUPPORTED = {".pdf", ".docx", ".xls", ".xlsx", ".xlsm", ".csv", ".txt", ".md"}
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer using only the context below.
If it does not answer the question, say so.
Do not invent facts. Be concise and helpful.

Context:
{context}

Question: {question}"""
)


def load_file(path: Path) -> list[Document]:
    """Load one supported file through the appropriate LangChain document loader."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        documents = PyPDFLoader(str(path), mode="page", extraction_mode="layout").load()
    elif ext == ".docx":
        documents = Docx2txtLoader(str(path)).load()
    elif ext in {".xls", ".xlsx", ".xlsm"}:
        documents = UnstructuredExcelLoader(str(path), mode="elements").load()
    else:
        documents = TextLoader(str(path), autodetect_encoding=True).load()

    result = []
    for document in documents:
        if not document.page_content.strip():
            continue
        document.metadata["source"] = str(path.resolve())
        # PyPDFLoader exposes zero-based pages; citations should be human-readable.
        if "page" in document.metadata:
            document.metadata["page"] = int(document.metadata["page"]) + 1
        result.append(document)
    return result


# This keeps the directory scan separate from file parsing so new formats are easy to add.
def find_document_files(documents_dir: Path) -> list[Path]:
    """Return supported files from the document directory and its subdirectories."""
    return sorted(
        path for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )

# 1. Load all supported documents and continue when an individual file cannot be read
def load_documents(documents_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in find_document_files(documents_dir):
        try:
            loaded = load_file(path)
            documents.extend(loaded)
            print(f"Loaded {path.name}: {len(loaded)} document element(s)")
        except Exception as error:
            print(f"Skipped {path.name}: {error}", file=sys.stderr)
    return documents


# 2. SPLIT INTO CHUNKS
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)


# 3. EMBED CHUNKS WITH OLLAMA
def create_embeddings(embedding_model: str) -> OllamaEmbeddings:
    """Create the local Ollama embedding client used by Chroma for indexing and search."""
    return OllamaEmbeddings(model=embedding_model)


# 4. STORE EMBEDDINGS IN CHROMA
def get_vector_store(database_dir: Path, embedding_model: str) -> Chroma:
    """Open the persistent Chroma collection backed by local Ollama embeddings."""
    return Chroma(
        collection_name="askmydocs",
        persist_directory=str(database_dir),
        embedding_function=create_embeddings(embedding_model),
    )


def store_chunks(chunks: list[Document], database_dir: Path, embedding_model: str) -> int:
    """Replace old chunks for each source and add the newly embedded chunks to Chroma."""
    if not chunks:
        return 0
    vector_store = get_vector_store(database_dir, embedding_model)
    sources = {chunk.metadata["source"] for chunk in chunks}
    for source in sources:
        vector_store.delete(where={"source": source})
    vector_store.add_documents(chunks)
    return len(chunks)

# This method orchestrates the ingestion pipeline: it loads documents, splits them into chunks, embeds them, and stores them in the Chroma database. It also handles rebuilding the database if specified.
def ingest(documents_dir: Path, database_dir: Path, embedding_model: str, rebuild: bool) -> None:
    """Run the ingestion pipeline: load documents, split chunks, embed, and store them."""
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")
    if rebuild and database_dir.exists():
        shutil.rmtree(database_dir)
    files = find_document_files(documents_dir)
    if not files:
        print(f"No supported documents found in {documents_dir}")
        return
    documents = load_documents(documents_dir)
    chunks = split_documents(documents)
    stored = store_chunks(chunks, database_dir, embedding_model)
    print(f"Done: stored {stored} chunks from {len(files)} file(s).")


# 5. RETRIEVE RELEVANT CHUNKS
def get_retriever(database_dir: Path, embedding_model: str, top_k: int):
    """Create a similarity-search retriever for the user's question."""
    if not database_dir.exists():
        raise FileNotFoundError("No index found. Run `uv run python main.py ingest` first.")
    return get_vector_store(database_dir, embedding_model).as_retriever(
        search_kwargs={"k": top_k}
    )


def format_context(documents: list[Document]) -> str:
    """Format retrieved chunks with source names before they are sent to the LLM."""
    return "\n\n---\n\n".join(
        f"Source: {Path(document.metadata['source']).name}\n{document.page_content}"
        for document in documents
    )


def format_sources(documents: list[Document]) -> list[str]:
    """Create concise, de-duplicated source citations for the CLI and browser UI."""
    sources = []
    for document in documents:
        label = Path(document.metadata["source"]).name
        if page := document.metadata.get("page"):
            label += f", page {page}"
        if sheet := document.metadata.get("sheet"):
            label += f", sheet {sheet}"
        if label not in sources:
            sources.append(label)
    return sources


# 6. GENERATE AN ANSWER WITH OLLAMA
def get_llm(model: str) -> ChatOllama:
    """Create the deterministic local Ollama chat model used for final answers."""
    return ChatOllama(model=model, temperature=0)


def answer(question: str, database_dir: Path, model: str, embedding_model: str, top_k: int) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and ask Ollama to answer only from their context."""
    documents = get_retriever(database_dir, embedding_model, top_k).invoke(question)
    if not documents:
        return "No relevant content was found in the index.", []

    # Format the context and apply the prompt template
    context = format_context(documents)
    prompt = PROMPT.format(context=context, question=question)

    # Query the LLM
    response = get_llm(model).invoke(prompt)

    return str(response.content), format_sources(documents)


def ask(question: str, database_dir: Path, model: str, embedding_model: str, top_k: int) -> None:
    response, sources = answer(question, database_dir, model, embedding_model, top_k)
    print(f"\nAnswer:\n{response}\n")
    print("Sources: " + "; ".join(sources))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about local documents with Ollama.")
    parser.add_argument("--documents-dir", type=Path, default=ROOT / "documents")
    parser.add_argument("--database-dir", type=Path, default=ROOT / ".rag_db")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1"))
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    parser.add_argument("--top-k", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("ingest", help="Read documents and create/update the local index")
    command.add_argument("--rebuild", action="store_true", help="Delete the existing index before ingesting")
    commands.add_parser("chat", help="Open an interactive question-and-answer session")
    command = commands.add_parser("ask", help="Ask one question and exit")
    command.add_argument("question")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        if args.command == "ingest":
            ingest(args.documents_dir.resolve(), args.database_dir.resolve(), args.embedding_model, args.rebuild)
        elif args.command == "ask":
            ask(args.question, args.database_dir.resolve(), args.model, args.embedding_model, args.top_k)
        else:
            print("Ask a question about your documents (type 'exit' to quit).")
            while (question := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
                if question:
                    ask(question, args.database_dir.resolve(), args.model, args.embedding_model, args.top_k)
    except (FileNotFoundError, ConnectionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
