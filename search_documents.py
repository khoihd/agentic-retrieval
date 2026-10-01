import re
from pathlib import Path
from google.genai import types
from utils import embed_text, cosine_similarity, get_client

def generate_grounded_answer(user_query, context_chunks):
    evidence = format_retrieval_context(context_chunks)
    prompt = f"""
Answer the user's question using only the evidence provided below.
Rules:
- Do not use information that is not supported by the evidence.
- Cite factual claims using the supporting SOURCE id in square brackets.
- If the evidence does not establish part of the answer, explicitly say so.
- Do not infer or invent missing company policy.

USER QUESTION:
{user_query}

EVIDENCE:
{evidence}
"""
    client = get_client()
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
            ),
        ),
    )

    return response.text

def format_retrieval_context(context_chunks):
    sections = []
    for chunk in context_chunks:
        sections.append(
            f"""SOURCE: {chunk["chunk_id"]}
            Document: {chunk["title"]}
            Section: {chunk["section"]} 
            Evidence:{chunk["text"]}"""
        )

    return "\n\n".join(sections)

def build_retrieval_context(original_query, retrieved_chunks, max_chars=4000):
    seen = set()
    unique_chunks = []

    for chunk in retrieved_chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        unique_chunks.append(chunk)

    reranked_chunks = rerank_chunks(
        original_query,
        unique_chunks,
    )

    selected_chunks = []
    used_chars = 0
    for chunk in reranked_chunks:
        chunk_size = len(chunk["text"])
        if used_chars + chunk_size > max_chars:
            continue

        selected_chunks.append(chunk)
        used_chars += chunk_size

    return selected_chunks

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


def load_document(file_path, document_id, title, category=None):
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


def chunk_document(document, max_chars=500, overlap_sentences=1):
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


document = load_document("travel_policy.txt", "id_1", "Travel Policy")
chunks = chunk_document(document)
embedded_chunks = embed_chunks(chunks)


def search_documents(query: str) -> str:
    """Search internal company documents for information relevant to a query."""
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


def split_into_sentences(text):
    # Simple learning-oriented splitter
    return re.split(r"(?<=[.!?])\s+", text.strip())


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


def split_into_sections(text):
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
