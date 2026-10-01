from typing import Literal
from pydantic import BaseModel


class MemoryRelationship(BaseModel):
    relationship: Literal["duplicate", "superseding", "conflicting", "related"]
    id: str
