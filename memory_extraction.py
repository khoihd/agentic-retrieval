from pydantic import BaseModel
from utils import get_client, MODEL


class ExtractedMemory(BaseModel):
    text: str
    memory_type: str
    scope: str


class MemoryExtractionResult(BaseModel):
    memories: list[ExtractedMemory]


def extract_memories(text: str) -> list[ExtractedMemory]:
    prompt = f"""

Extract information from the input that is worth storing in long-term memory.

Store information that:
- is likely to be useful in future interactions
- is reasonably durable rather than temporary
- describes user preferences, decisions, or project-specific facts
- can be represented as a clear, self-contained fact

Do not store:
- conversational filler
- questions that contain no useful durable information
- temporary task state that is only useful in the current interaction
- information unlikely to matter in future interactions

Guidelines:
- Each memory should contain one atomic fact when possible.
- Rewrite memories as clear, self-contained statements.
- A single input may produce zero, one, or multiple memories.
- Choose an appropriate memory_type.
- Choose an appropriate scope.

Scope rules:
- Use "user" for information about the user's general preferences,
  habits, or personal information.
- For project-specific information, derive a normalized project scope.
  For example:
  "Project X" -> "project_x"
  "Project Alpha" -> "project_alpha"
- Do not use "global" for user-specific or project-specific information.

INPUT:
{text}
"""
    response = get_client().beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=MemoryExtractionResult,
        temperature=0,
    )

    result = response.choices[0].message.parsed

    return result.memories
