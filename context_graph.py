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
import networkx as nx


# ─────────────────────────────────────────────────────────────
# Node factory helpers
# ─────────────────────────────────────────────────────────────

def _node(node_type: str, node_id: str, **attrs) -> dict:
    return {"id": node_id, "type": node_type, "created_at": datetime.utcnow().isoformat(), **attrs}


def user_node(user_id: str, name: str, role: str, grade: str = "", school: str = "") -> dict:
    return _node("User", user_id, name=name, role=role, grade=grade, school=school)


def goal_node(goal_id: str, title: str, deadline: str, progress: int = 0,
              status: str = "active", description: str = "") -> dict:
    return _node("Goal", goal_id, title=title, deadline=deadline,
                 progress=progress, status=status, description=description)


def course_node(course_id: str, name: str, instructor: str = "",
                schedule: str = "", grade_earned: str = "") -> dict:
    return _node("Course", course_id, name=name, instructor=instructor,
                 schedule=schedule, grade_earned=grade_earned)


def assignment_node(asgn_id: str, title: str, due_date: str,
                    status: str = "pending", instructions: str = "",
                    course_id: str = "") -> dict:
    return _node("Assignment", asgn_id, title=title, due_date=due_date,
                 status=status, instructions=instructions, course_id=course_id)


def screen_node(screen_id: str, name: str, description: str = "") -> dict:
    return _node("Screen", screen_id, name=name, description=description)


def conversation_turn_node(turn_id: str, role: str, content: str,
                            intent: str = "", entities: list[str] | None = None) -> dict:
    return _node("ConversationTurn", turn_id, role=role, content=content,
                 intent=intent, entities=entities or [],
                 timestamp=datetime.utcnow().isoformat())


def resource_node(res_id: str, title: str, url: str = "", res_type: str = "article") -> dict:
    return _node("Resource", res_id, title=title, url=url, res_type=res_type)


# ─────────────────────────────────────────────────────────────
# Context Graph class
# ─────────────────────────────────────────────────────────────

class ContextGraph:
    """
    In-memory directed graph that stores and exposes all context
    relevant to one user session.  In a multi-tenant production
    environment, the state dict can be serialised to/from Neo4j
    or Redis per session.
    """

    def __init__(self):
        self.g: nx.DiGraph = nx.DiGraph()

    # ── mutation ────────────────────────────────────────────

    def add_node(self, data: dict) -> None:
        nid = data["id"]
        self.g.add_node(nid, **data)

    def add_edge(self, src: str, dst: str, rel: str, **attrs) -> None:
        self.g.add_edge(src, dst, rel=rel, **attrs)

    def update_node(self, node_id: str, **attrs) -> None:
        if node_id in self.g.nodes:
            self.g.nodes[node_id].update(attrs)

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

    # ── subgraph builder (the key runtime primitive) ─────────

    def build_relevant_subgraph(self, user_id: str, max_turns: int = 4) -> dict:
        """
        Query the graph and assemble only the context nodes that are
        RELEVANT for the current conversational turn.

        Returns a plain dict — this is what gets serialised into
        the LLM prompt.  The selection logic here is the core
        advantage over naïve prompt-stuffing.
        """
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