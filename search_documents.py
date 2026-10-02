import re
from pathlib import Path
from google.genai import types
from utils import embed_text, cosine_similarity


def split_into_sentences(text):
    # Simple learning-oriented splitter
    return re.split(r"(?<=[.!?])\s+", text.strip())


def rerank_chunks(original_query, chunks):
    query_embedding = embed_text(original_query)
    reranked_chunks = []

    for chunk in chunks:
        chunk_embedding = embed_text(chunk["text"])
        rerank_score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        reranked_chunk = chunk.copy()
        reranked_chunk["rerank_score"] = rerank_score
        reranked_chunks.append(reranked_chunk)

    reranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked_chunks


def search_chunks(query, chunks, top_k=3):
    query_embedding = embed_text(query)
    results = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        results.append({"chunk": chunk, "score": score})

    results.sort(key=lambda result: result["score"], reverse=True)

    return results[:top_k]


# each chunk, which is a mini-paragraph, has max_chars, with overlap_sentences
def chunk_section(section_text, max_chars=500, overlap_sentences=1):
    sentences = split_into_sentences(section_text)
    chunks = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if len(candidate) <= max_chars:
            current.append(sentence)
        else:
            if current:
                chunks.append(" ".join(current))

            # Keep a little context from previous chunk
            if overlap_sentences > 0 and current:
                current = current[-overlap_sentences:]
            else:
                current = []

            current.append(sentence)

    # append the last current, list of sentences
    if current:
        chunks.append(" ".join(current))

    return chunks


def split_into_sections(text: str) -> list[dict]:
    sections = []
    current_section = "Introduction"
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("#"):
            if current_lines:
                sections.append(
                    {"section": current_section, "text": " ".join(current_lines)}
                )

            current_section = line.lstrip("#").strip()

            current_lines = []

        elif line:
            current_lines.append(line)

    if current_lines:
        sections.append({"section": current_section, "text": " ".join(current_lines)})

    return sections


def format_document_retrieval_context(context_chunks):
    sections = []
    for chunk in context_chunks:
        sections.append(
            f"""SOURCE: {chunk["chunk_id"]}
            Document: {chunk["title"]}
            Section: {chunk["section"]} 
            Evidence:{chunk["text"]}"""
        )

    return "\n\n".join(sections)


def get_search_documents_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="search_documents",
        description=(
            "Search internal company documents for information relevant to a query."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the information needed.",
                }
            },
            "required": ["query"],
        },
    )


def load_document(file_path: str, document_id: str, title: str, category=None) -> dict:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    return {
        "document_id": document_id,
        "title": title,
        "category": category,
        "source_path": str(path),
        "text": text,
    }


def chunk_document(document: dict, max_chars: int = 500, overlap_sentences: int = 1) -> list[dict]:
    chunks = []
    chunk_index = 0
    sections = split_into_sections(document["text"])

    for section in sections:
        section_chunks = chunk_section(
            section["text"], max_chars=max_chars, overlap_sentences=overlap_sentences
        )

        for text in section_chunks:
            chunk = {
                "chunk_id": (f"{document['document_id']}_{chunk_index:04d}"),
                "chunk_index": chunk_index,
                "document_id": document["document_id"],
                "title": document["title"],
                "category": document.get("category"),
                "source_path": document.get("source_path"),
                "section": section["section"],
                "text": text,
            }

            chunks.append(chunk)

            chunk_index += 1

    return chunks


def embed_chunks(chunks):
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])

    return chunks


def search_documents(query: str) -> str:
    """Search internal company documents for information relevant to a query."""
    
    document = load_document("travel_policy.txt", "id_1", "Travel Policy")
    chunks = chunk_document(document)
    embedded_chunks = embed_chunks(chunks)
    results = search_chunks(query=query, chunks=embedded_chunks, top_k=1)

    documents = []
    for result in results:
        chunk = result["chunk"]
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": float(result["score"]),
            }
        )

    return documents
