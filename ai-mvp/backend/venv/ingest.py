"""
Document ingestion pipeline for RAG system.
Handles PDF, PPTX, DOCX, TXT files → chunks → embeddings → ChromaDB.
Uses sentence-transformers for local embeddings (no API key needed).
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import tiktoken


# Configuration
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "study_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality local embedding model
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Initialize clients
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)

# Local embedding model (downloads on first run, then cached)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Tokenizer for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")


def get_collection():
    """Get or create the ChromaDB collection."""
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by token count."""
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        if i + chunk_size >= len(tokens):
            break

    return chunks


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page numbers."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": i + 1,
                "source_type": "pdf"
            })
    return pages


def extract_text_from_pptx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PPTX with slide numbers."""
    prs = Presentation(file_path)
    slides = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    texts.append(para.text)
        full_text = "\n".join(texts)
        if full_text.strip():
            slides.append({
                "text": full_text,
                "slide": i + 1,
                "source_type": "pptx"
            })
    return slides


def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX."""
    doc = DocxDocument(file_path)
    paragraphs = []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            paragraphs.append({
                "text": para.text,
                "paragraph": i + 1,
                "source_type": "docx"
            })
    return paragraphs


def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from TXT/MD file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [{
        "text": text,
        "source_type": "txt"
    }]


def extract_text(file_path: str) -> List[Dict[str, Any]]:
    """Route to appropriate extractor based on file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.pptx':
        return extract_text_from_pptx(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.txt', '.md']:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_file_hash(file_path: str) -> str:
    """Generate SHA256 hash of file for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local sentence-transformers model."""
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def ingest_file(file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest a single document into the vector store.
    Returns metadata about the ingestion.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_hash = get_file_hash(str(file_path))
    doc_id = doc_id or file_hash[:12]

    collection = get_collection()

    # Check if already ingested
    existing = collection.get(ids=[doc_id])
    if existing['ids']:
        return {
            "status": "skipped",
            "doc_id": doc_id,
            "message": "Document already ingested"
        }

    # Extract text
    pages = extract_text(str(file_path))

    # Chunk each page/slide/section
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for page_idx, page in enumerate(pages):
        chunks = chunk_text(page["text"])

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_p{page_idx}_c{chunk_idx}"
            metadata = {
                "doc_id": doc_id,
                "filename": file_path.name,
                "chunk_index": chunk_idx,
                "source_type": page["source_type"],
                "text": chunk[:500],  # Store preview in metadata
            }

            # Add source-specific location info
            if "page" in page:
                metadata["page"] = page["page"]
            if "slide" in page:
                metadata["slide"] = page["slide"]
            if "paragraph" in page:
                metadata["paragraph"] = page["paragraph"]

            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    # Generate embeddings
    embeddings = generate_embeddings(all_chunks)

    # Store in ChromaDB
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file_path.name,
        "chunks": len(all_chunks),
        "pages_slides": len(pages)
    }


def ingest_directory(directory: str) -> List[Dict[str, Any]]:
    """Ingest all supported files in a directory."""
    supported_ext = {'.pdf', '.pptx', '.docx', '.txt', '.md'}
    results = []

    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in supported_ext:
            try:
                result = ingest_file(str(file_path))
                results.append(result)
                print(f"Ingested: {file_path.name} ({result['chunks']} chunks)")
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": file_path.name,
                    "error": str(e)
                })
                print(f"Error ingesting {file_path.name}: {e}")

    return results


def list_documents() -> List[Dict[str, Any]]:
    """List all ingested documents with metadata."""
    collection = get_collection()
    results = collection.get()

    # Group by doc_id
    docs = {}
    for metadata in results.get('metadatas', []):
        doc_id = metadata.get('doc_id')
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": metadata.get('filename'),
                "source_type": metadata.get('source_type'),
                "chunks": 0
            }
        docs[doc_id]["chunks"] += 1

    return list(docs.values())


def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document and all its chunks from the vector store."""
    collection = get_collection()

    # Find all chunk IDs for this doc_id
    results = collection.get(where={"doc_id": doc_id})
    chunk_ids = results.get('ids', [])

    if not chunk_ids:
        return {"status": "not_found", "doc_id": doc_id}

    collection.delete(ids=chunk_ids)
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": len(chunk_ids)}


def search_similar(query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for similar chunks to the query."""
    collection = get_collection()

    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]

    # Build where filter
    where = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=['documents', 'metadatas', 'distances']
    )

    # Format results
    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i],
            "similarity": 1 - results['distances'][0][i]  # cosine similarity
        })

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        results = ingest_directory(path)
    else:
        results = [ingest_file(path)]

    for r in results:
        print(r)