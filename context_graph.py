"""
context_graph.py
----------------
Core context graph implementation using NetworkX.
Models the multi-dimensional context of a student inside an
education/career-readiness SaaS platform.

Node Types
----------
UserNode         – identity, role, grade level
GoalNode         – career/academic goal with progress
CourseNode       – enrolled course
AssignmentNode   – specific task tied to a course or goal
ScreenNode       – current platform screen / workflow state
ConversationTurn – single message in the chat history
ResourceNode     – article, video, or tool linked to a goal

Relationship Types
------------------
HAS_GOAL          User ──► Goal
ENROLLED_IN       User ──► Course
WORKING_ON        User ──► Assignment   (active)
CURRENTLY_ON      User ──► Screen
HAD_TURN          User ──► ConversationTurn
REQUIRES          Goal ──► Assignment
PART_OF           Assignment ──► Course
REFERENCES        ConversationTurn ──► Goal | Course | Assignment
SUGGESTED         Goal ──► Resource
RELATED_TO        Goal ──► Goal         (cross-domain links)
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Optional
import numpy as np
import networkx as nx

# ─────────────────────────────────────────────────────────────
# Embedding model — lazy-loaded once, shared across all graph
# instances in the process.
#
# Model: all-MiniLM-L6-v2  (~90 MB, 384-dim, CPU-friendly)
# Falls back to None if sentence-transformers is not installed;
# extract_entities then degrades to token-overlap matching.
# ─────────────────────────────────────────────────────────────

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embed_model():
    """Lazy-load the embedding model once and cache it globally."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[EMBED] Loading '{_EMBED_MODEL_NAME}' … (one-time, ~2 s)")
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
        print("[EMBED] Model ready.")
    except ImportError:
        print("[WARN] sentence-transformers not installed. "
              "Falling back to token-overlap entity extraction.")
        _EMBED_MODEL = None
    return _EMBED_MODEL


def _embed_text(text: str) -> np.ndarray | None:
    """Embed a string to a unit-norm vector, or return None on failure."""
    model = _get_embed_model()
    if model is None or not text.strip():
        return None
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def _node_label(data: dict) -> str:
    """Compose the text we embed for a graph node."""
    parts = filter(None, [
        data.get("title") or data.get("name") or "",
        data.get("description", ""),
        data.get("instructions", "")[:120],   # first 120 chars of instructions
    ])
    return " ".join(parts).strip()



# ─────────────────────────────────────────────────────────────
# Pydantic node schemas
#
# Using Pydantic BaseModel instead of plain dicts gives us:
#   - Field validation and type coercion at construction time.
#   - A clear, self-documenting schema for each node type.
#   - model_dump() for easy serialisation back to dicts for NetworkX.
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Literal


class BaseNode(BaseModel):
    id: str
    type: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UserNode(BaseNode):
    type: Literal["User"] = "User"
    name: str
    role: Literal["student", "parent", "counselor"]
    grade: str = ""
    school: str = ""


class GoalNode(BaseNode):
    type: Literal["Goal"] = "Goal"
    title: str
    deadline: str
    progress: int = Field(default=0, ge=0, le=100)
    status: str = "active"
    description: str = ""


class CourseNode(BaseNode):
    type: Literal["Course"] = "Course"
    name: str
    instructor: str = ""
    schedule: str = ""
    grade_earned: str = ""


class AssignmentNode(BaseNode):
    type: Literal["Assignment"] = "Assignment"
    title: str
    due_date: str
    status: str = "pending"
    instructions: str = ""
    course_id: str = ""


class ScreenNode(BaseNode):
    type: Literal["Screen"] = "Screen"
    name: str
    description: str = ""


class ConversationTurnNode(BaseNode):
    type: Literal["ConversationTurn"] = "ConversationTurn"
    role: str
    content: str
    intent: str = ""
    entities: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ResourceNode(BaseNode):
    type: Literal["Resource"] = "Resource"
    title: str
    url: str = ""
    res_type: str = "article"


# ─────────────────────────────────────────────────────────────
# Factory helpers  (return dicts for NetworkX compatibility)
# ─────────────────────────────────────────────────────────────

def user_node(user_id: str, name: str, role: str, grade: str = "", school: str = "") -> dict:
    return UserNode(id=user_id, name=name, role=role, grade=grade, school=school).model_dump()


def goal_node(goal_id: str, title: str, deadline: str, progress: int = 0,
              status: str = "active", description: str = "") -> dict:
    return GoalNode(id=goal_id, title=title, deadline=deadline,
                    progress=progress, status=status, description=description).model_dump()


def course_node(course_id: str, name: str, instructor: str = "",
                schedule: str = "", grade_earned: str = "") -> dict:
    return CourseNode(id=course_id, name=name, instructor=instructor,
                      schedule=schedule, grade_earned=grade_earned).model_dump()


def assignment_node(asgn_id: str, title: str, due_date: str,
                    status: str = "pending", instructions: str = "",
                    course_id: str = "") -> dict:
    return AssignmentNode(id=asgn_id, title=title, due_date=due_date,
                          status=status, instructions=instructions,
                          course_id=course_id).model_dump()


def screen_node(screen_id: str, name: str, description: str = "") -> dict:
    return ScreenNode(id=screen_id, name=name, description=description).model_dump()


def conversation_turn_node(turn_id: str, role: str, content: str,
                            intent: str = "", entities: list[str] | None = None) -> dict:
    return ConversationTurnNode(id=turn_id, role=role, content=content,
                                intent=intent, entities=entities or []).model_dump()


def resource_node(res_id: str, title: str, url: str = "", res_type: str = "article") -> dict:
    return ResourceNode(id=res_id, title=title, url=url, res_type=res_type).model_dump()



# ─────────────────────────────────────────────────────────────
# Context Graph class
# ─────────────────────────────────────────────────────────────

class ContextGraph:
    """
    In-memory directed graph that stores and exposes all context
    relevant to one user session, scoped to a single tenant.

    tenant_id is the top-level isolation boundary: in production each
    tenant gets either a dedicated Neo4j database or a namespace-prefixed
    keyspace in a shared store. Passing it here makes that contract
    explicit at the data layer rather than just in documentation.

    In a multi-tenant production environment the state dict can be
    serialised to/from Neo4j or Redis per session.
    """

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id: str = tenant_id
        self.g: nx.DiGraph = nx.DiGraph()
        # node_id → unit-norm embedding vector (np.float32)
        self._embed_store: dict[str, np.ndarray] = {}
        # node_id → text used to produce the embedding (for cache validation)
        self._node_text: dict[str, str] = {}

    # ── mutation ────────────────────────────────────────────

    def add_node(self, data: dict) -> None:
        nid = data["id"]
        self.g.add_node(nid, **data)
        # Embed the node label at creation time so queries are instant.
        text = _node_label(data)
        if text:
            vec = _embed_text(text)
            if vec is not None:
                self._embed_store[nid] = vec
                self._node_text[nid] = text

    def add_edge(self, src: str, dst: str, rel: str, **attrs) -> None:
        self.g.add_edge(src, dst, rel=rel, **attrs)

    def update_node(self, node_id: str, **attrs) -> None:
        if node_id in self.g.nodes:
            self.g.nodes[node_id].update(attrs)
            # Invalidate the cached embedding if the label-forming fields changed.
            if any(k in attrs for k in ("title", "name", "description", "instructions")):
                new_text = _node_label(dict(self.g.nodes[node_id]))
                if new_text != self._node_text.get(node_id, ""):
                    vec = _embed_text(new_text)
                    if vec is not None:
                        self._embed_store[node_id] = vec
                        self._node_text[node_id] = new_text
                    else:
                        self._embed_store.pop(node_id, None)
                        self._node_text.pop(node_id, None)

    # ── primitive queries ────────────────────────────────────

    def get_node(self, node_id: str) -> dict:
        return dict(self.g.nodes.get(node_id, {}))

    def get_neighbors(self, node_id: str, rel: str | None = None) -> list[dict]:
        result = []
        for _, dst, data in self.g.out_edges(node_id, data=True):
            if rel is None or data.get("rel") == rel:
                result.append(self.get_node(dst))
        return result

    def nodes_by_type(self, node_type: str) -> list[dict]:
        return [dict(d) for _, d in self.g.nodes(data=True) if d.get("type") == node_type]

    # ── semantic / proximity queries ─────────────────────────

    def get_active_goals(self, user_id: str) -> list[dict]:
        return [n for n in self.get_neighbors(user_id, "HAS_GOAL")
                if n.get("status") == "active"]

    def get_current_screen(self, user_id: str) -> dict | None:
        screens = self.get_neighbors(user_id, "CURRENTLY_ON")
        return screens[-1] if screens else None

    def get_recent_turns(self, user_id: str, k: int = 6) -> list[dict]:
        turns = self.get_neighbors(user_id, "HAD_TURN")
        turns.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        return turns[:k]

    def get_active_assignment(self, user_id: str) -> dict | None:
        assignments = self.get_neighbors(user_id, "WORKING_ON")
        pending = [a for a in assignments if a.get("status") in ("pending", "in_progress")]
        if not pending:
            return None
        pending.sort(key=lambda a: a.get("due_date", "9999-99-99"))
        return pending[0]

    def get_goal_resources(self, goal_id: str) -> list[dict]:
        return self.get_neighbors(goal_id, "SUGGESTED")

    def get_inferred_intent(self, user_id: str) -> str:
        """Return the intent label from the most recent conversation turn."""
        turns = self.get_recent_turns(user_id, k=1)
        return turns[0].get("intent", "general_inquiry") if turns else "general_inquiry"

    # ── semantic entity search ────────────────────────────────

    def semantic_search(
        self, query: str, top_k: int = 5, threshold: float = 0.40
    ) -> list[tuple[str, float]]:
        """
        Embed *query* and return the top-k node IDs whose stored embedding
        has cosine similarity ≥ threshold.

        Because all embeddings are unit-normalised (normalize_embeddings=True),
        cosine similarity = dot product — no extra division needed.

        Returns a list of (node_id, score) tuples sorted by score descending.
        Falls back to [] if no embeddings are stored (e.g. model not loaded).
        """
        if not self._embed_store:
            return []

        query_vec = _embed_text(query)
        if query_vec is None:
            return []

        # Stack all stored vectors into a matrix for one batched dot product.
        ids = list(self._embed_store.keys())
        matrix = np.stack([self._embed_store[nid] for nid in ids])  # (N, D)
        scores = matrix @ query_vec                                   # (N,)

        results = [
            (nid, float(score))
            for nid, score in zip(ids, scores)
            if score >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── history pruning (prevents unbounded graph growth) ──────

    def prune_old_turns(self, user_id: str, max_turns: int = 20) -> int:
        """
        Sliding-window pruner: keeps only the `max_turns` most-recent
        ConversationTurn nodes for a user and removes everything older.

        In production you would archive evicted turns to a cold store
        (e.g. append to a PostgreSQL jsonb column) before removal so that
        long-term memory can be reconstructed via summarisation.

        Returns the number of nodes evicted.
        """
        turns = self.get_neighbors(user_id, "HAD_TURN")
        if len(turns) <= max_turns:
            return 0

        # Sort oldest-first; evict everything beyond the window
        turns.sort(key=lambda t: t.get("timestamp", ""))
        evict = turns[: len(turns) - max_turns]
        for turn in evict:
            tid = turn["id"]
            # Remove all edges first, then the node
            self.g.remove_edges_from(list(self.g.in_edges(tid)) + list(self.g.out_edges(tid)))
            self.g.remove_node(tid)

        return len(evict)

    # ── subgraph builder (the key runtime primitive) ─────────

    def build_relevant_subgraph(self, user_id: str, max_turns: int = 4,
                                tenant_id: str | None = None) -> dict:
        """
        Query the graph and assemble only the context nodes that are
        RELEVANT for the current conversational turn.

        tenant_id, when provided, is validated against self.tenant_id to
        prevent cross-tenant data leakage — the first guard in a
        multi-tenant production pipeline.

        Returns a plain dict — this is what gets serialised into
        the LLM prompt.  The selection logic here is the core
        advantage over naive prompt-stuffing.
        """
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise PermissionError(
                f"Tenant mismatch: graph belongs to '{self.tenant_id}', "
                f"but caller passed '{tenant_id}'.")

        user = self.get_node(user_id)
        screen = self.get_current_screen(user_id)
        active_goals = self.get_active_goals(user_id)
        active_asgn = self.get_active_assignment(user_id)
        recent_turns = self.get_recent_turns(user_id, k=max_turns)

        # Enrich goals with their nearest resources
        goals_enriched = []
        for g in active_goals:
            resources = self.get_goal_resources(g["id"])
            goals_enriched.append({**g, "resources": resources})

        # Infer focal goal: the one referenced in recent turns (if any)
        referenced_ids = set()
        for t in recent_turns:
            referenced_ids.update(t.get("entities", []))
        focal_goal = next((g for g in active_goals if g["id"] in referenced_ids), None)

        return {
            "user": user,
            "current_screen": screen,
            "active_goals": goals_enriched,
            "focal_goal": focal_goal,
            "active_assignment": active_asgn,
            "recent_conversation": recent_turns,
            "inferred_intent": self.get_inferred_intent(user_id),
        }

    # ── snapshot / debug ─────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "nodes": [dict(d) for _, d in self.g.nodes(data=True)],
            "edges": [{"src": u, "dst": v, "rel": d.get("rel")}
                      for u, v, d in self.g.edges(data=True)],
        }