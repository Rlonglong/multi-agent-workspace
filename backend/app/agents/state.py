# State definitions for multi‑stage workflow
from typing import Annotated, TypedDict, Any, List, Dict, Optional, Literal
try:
    from langchain_core.messages import BaseMessage
except ImportError:
    from langchain.schema import BaseMessage  # fallback for newer versions
from langgraph.graph.message import add_messages

# Helper functions (kept from previous version)

def update_artifacts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new artifacts into the existing artifact dictionary."""
    if not existing:
        return new
    updated = existing.copy()
    updated.update(new)
    return updated


def update_agents(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Overwrite the existing agent formation when the PM generates a new one, or leave as existing."""
    if new is not None and len(new) > 0:
        return new
    return existing


class AgentConfig(TypedDict, total=False):
    role: str
    model: str
    prompt: str
    apiKey: str
    temperature: Optional[float]
    top_p: Optional[float]


class CollapsibleCodeBlock(TypedDict, total=False):
    language: str          # e.g. "python", "tsx", "sql"
    content: str           # raw source code
    collapsible: bool      # always true for generated blocks
    metadata: Optional[Dict[str, Any]]


class WorkspaceState(TypedDict):
    # Core LLM message history, utilizing LangGraph's built‑in message reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Dynamically generated Swarm agent roster
    agent_configs: Annotated[List[AgentConfig], update_agents]

    # Tracking which agent processed last and which is next
    sender: str
    next: str

    # Artifacts (filename -> content) produced by agents
    artifacts: Annotated[Dict[str, Any], update_artifacts]

    # ----- New fields for the 3‑stage UI workflow -----
    stage: Literal["discovery", "implementation", "agent_config", "execution"]
    sidebar_visible: bool
    guideline: str
    guideline_editable: bool
    code_blocks: List[CollapsibleCodeBlock]
    extra: Optional[Dict[str, Any]]
    # New field to store message edit history (branching)
    message_branches: Dict[str, List[str]]
    execution_started: bool
    execution_queue: List[str]
    execution_cursor: int
