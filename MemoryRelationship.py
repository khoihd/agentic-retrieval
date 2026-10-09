from typing import Literal

from pydantic import BaseModel


class MemoryRelationship(BaseModel):
    relationship: Literal["duplicate", "superseding", "conflicting", "related"]
    id: str

class MemoryRelationshipList(BaseModel):
    relationships: list[MemoryRelationship]
