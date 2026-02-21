"""
langgraph_flow.py
-----------------
LangGraph state machine that orchestrates the context-graph–driven
conversational assistant.

Graph nodes (pipeline stages)
──────────────────────────────
1. ingest_message      – receive user text; classify intent; extract entities
2. update_graph        – add new ConversationTurn node; update WORKING_ON / screen
3. query_graph         – call ContextGraph.build_relevant_subgraph()
4. build_prompt        – render a tight, graph-informed system prompt
5. generate_response   – call the LLM
6. post_process        – strip artifacts, add citation metadata
7. END

The same pipeline runs in two modes:
  - "graph"    → uses the context graph (full pipeline)
  - "baseline" → skips steps 2-3 and uses a flat prompt (for comparison)
"""

from __future__ import annotations
import json
import os
import uuid
from typing import Any, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from context_graph import (
    ContextGraph,
    conversation_turn_node,
    screen_node,
)

# ─────────────────────────────────────────────────────────────
# Shared LangGraph State
# ─────────────────────────────────────────────────────────────

class AssistantState(TypedDict):
    # ── inputs ──────────────────────────────────────────────
    user_id: str
    user_message: str
    mode: Literal["graph", "baseline"]           # "graph" | "baseline"

    # ── context graph (only used in graph mode) ──────────────
    context_graph: ContextGraph | None
    subgraph_context: dict | None                # output of build_relevant_subgraph

    # ── prompt construction ──────────────────────────────────
    system_prompt: str
    conversation_history: list[dict]             # flat [{role, content}] for LLM

    # ── llm outputs ──────────────────────────────────────────
    ai_response: str
    inferred_intent: str
    referenced_entities: list[str]


# ─────────────────────────────────────────────────────────────
# Intent / entity extraction (lightweight; swap for NER model)
# ─────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {
    "assignment_help":    ["assignment", "essay", "due", "submit", "homework", "task"],
    "goal_progress":      ["goal", "progress", "track", "update", "complete", "milestone"],
    "course_inquiry":     ["course", "class", "grade", "teacher", "instructor", "schedule"],
    "resource_request":   ["resource", "link", "video", "article", "example", "guide"],
    "deadline_check":     ["deadline", "when", "due date", "time left", "overdue"],
    "general_inquiry":    [],  # fallback
}


def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in t for k in keywords):
            return intent
    return "general_inquiry"


def extract_entities(text: str, ctx_graph: ContextGraph | None, user_id: str) -> list[str]:
    """Return node IDs whose label/title appears in the user message."""
    if ctx_graph is None:
        return []
    found = []
    for nid, data in ctx_graph.g.nodes(data=True):
        label = (data.get("title") or data.get("name") or "").lower()
        if label and label in text.lower():
            found.append(nid)
    return found


# ─────────────────────────────────────────────────────────────
# Pipeline node functions
# ─────────────────────────────────────────────────────────────

def ingest_message(state: AssistantState) -> AssistantState:
    """Classify intent and extract entity references from the user message."""
    intent = classify_intent(state["user_message"])
    entities = extract_entities(state["user_message"],
                                state.get("context_graph"),
                                state["user_id"])
    return {**state, "inferred_intent": intent, "referenced_entities": entities}


def update_graph(state: AssistantState) -> AssistantState:
    """
    Persist the new ConversationTurn into the graph and refresh the
    CURRENTLY_ON screen if the UI layer reported a navigation event.
    This step is SKIPPED in baseline mode.
    """
    if state["mode"] != "graph" or state.get("context_graph") is None:
        return state

    cg: ContextGraph = state["context_graph"]
    turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    turn = conversation_turn_node(
        turn_id=turn_id,
        role="user",
        content=state["user_message"],
        intent=state["inferred_intent"],
        entities=state["referenced_entities"],
    )
    cg.add_node(turn)
    cg.add_edge(state["user_id"], turn_id, rel="HAD_TURN")

    # Wire turn → referenced entities
    for eid in state["referenced_entities"]:
        cg.add_edge(turn_id, eid, rel="REFERENCES")

    return {**state, "context_graph": cg}


def query_graph(state: AssistantState) -> AssistantState:
    """
    Pull the relevant subgraph for this turn.
    SKIPPED in baseline mode.
    """
    if state["mode"] != "graph" or state.get("context_graph") is None:
        return state

    cg: ContextGraph = state["context_graph"]
    subgraph = cg.build_relevant_subgraph(state["user_id"])
    return {**state, "subgraph_context": subgraph}


# ── Prompt builders ──────────────────────────────────────────

def _render_graph_prompt(sub: dict) -> str:
    """Convert the subgraph dict into a concise, structured system prompt."""
    user = sub["user"]
    screen = sub.get("current_screen") or {}
    goals = sub.get("active_goals") or []
    asgn = sub.get("active_assignment") or {}
    turns = sub.get("recent_conversation") or []
    intent = sub.get("inferred_intent", "general_inquiry")

    # Recent conversation (last N turns, oldest first)
    history_text = "\n".join(
        f"  [{t['role'].upper()}] {t['content']}" for t in reversed(turns)
    )

    # Active goals summary
    goals_text = ""
    for g in goals:
        resources = g.get("resources", [])
        res_titles = ", ".join(r["title"] for r in resources) if resources else "none"
        goals_text += (
            f"  • {g['title']} (progress: {g['progress']}%, deadline: {g['deadline']}, "
            f"resources: {res_titles})\n"
        )

    # Active assignment
    asgn_text = ""
    if asgn:
        asgn_text = (
            f"  Title: {asgn.get('title')}\n"
            f"  Due: {asgn.get('due_date')}\n"
            f"  Status: {asgn.get('status')}\n"
            f"  Instructions: {asgn.get('instructions', 'N/A')}\n"
        )

    return f"""You are an AI co-pilot inside an education/career-readiness platform.

=== USER CONTEXT (from Context Graph) ===
Name: {user.get('name')}  |  Role: {user.get('role')}  |  Grade: {user.get('grade')}
Current Screen: {screen.get('name', 'unknown')} — {screen.get('description', '')}
Inferred Intent: {intent}

=== ACTIVE GOALS ===
{goals_text or '  (none)'}
=== ACTIVE ASSIGNMENT ===
{asgn_text or '  (none)'}
=== RECENT CONVERSATION ===
{history_text or '  (no prior turns)'}

Instructions:
- Address the student by name where natural.
- Anchor your answer to their specific goal(s), assignment, and platform screen.
- Reference deadlines, progress percentages, or resource titles when relevant.
- Be encouraging but concise. Do not invent information not present above.
"""


def _render_baseline_prompt(user_message: str) -> str:
    """Flat prompt with no structured context — the naive baseline."""
    return (
        "You are a helpful educational assistant. "
        "Answer student questions about their coursework and goals as best you can."
    )


def build_prompt(state: AssistantState) -> AssistantState:
    if state["mode"] == "graph" and state.get("subgraph_context"):
        sys_prompt = _render_graph_prompt(state["subgraph_context"])
    else:
        sys_prompt = _render_baseline_prompt(state["user_message"])
    return {**state, "system_prompt": sys_prompt}


# ── LLM call ─────────────────────────────────────────────────

def _build_llm():
    """Return Gemini as primary LLM; fall back to HuggingFace if unavailable."""
    gemini_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token   = os.environ.get("HF_TOKEN", "")

    if gemini_key:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=gemini_key,
            )
        except Exception as e:
            print(f"[WARN] Gemini init failed ({e}); trying HuggingFace fallback…")

    if hf_token:
        endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)

    raise RuntimeError(
        "No LLM available. Set GOOGLE_API_KEY or HF_TOKEN environment variables."
    )


def generate_response(state: AssistantState) -> AssistantState:
    history = state.get("conversation_history") or []

    # Build a simple role/content message list for both providers
    chat_messages = [{"role": "system", "content": state["system_prompt"]}]
    for turn in history:
        if turn["role"] == "user":
            chat_messages.append({"role": "user", "content": turn["content"]})
    chat_messages.append({"role": "user", "content": state["user_message"]})

    gemini_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token   = os.environ.get("HF_TOKEN", "")

    # ── Primary: Google Gemini ──────────────────────────────────
    if gemini_key:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=gemini_key,
            )
            lc_messages = [SystemMessage(content=state["system_prompt"])]
            for turn in history:
                if turn["role"] == "user":
                    lc_messages.append(HumanMessage(content=turn["content"]))
            lc_messages.append(HumanMessage(content=state["user_message"]))
            response = llm.invoke(lc_messages)
            return {**state, "ai_response": response.content}
        except Exception as e:
            print(f"[WARN] Gemini failed ({type(e).__name__}); using HuggingFace fallback…")

    # ── Fallback: HuggingFace Inference API (direct, with retry) ──
    if hf_token:
        import time
        client = InferenceClient(token=hf_token)
        hf_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct",
        ]
        last_err = None
        for model_id in hf_models:
            for attempt in range(3):
                try:
                    result = client.chat_completion(
                        model=model_id,
                        messages=chat_messages,
                        max_tokens=512,
                    )
                    return {**state, "ai_response": result.choices[0].message.content}
                except Exception as hf_err:
                    last_err = hf_err
                    wait = 2 ** attempt
                    print(f"[WARN] HF {model_id} attempt {attempt+1} failed ({type(hf_err).__name__}); retrying in {wait}s…")
                    time.sleep(wait)
            print(f"[WARN] All retries exhausted for {model_id}; trying next model…")

        raise RuntimeError(f"All HuggingFace models failed. Last error: {last_err}")

    raise RuntimeError("No LLM available. Set GOOGLE_API_KEY or HF_TOKEN.")





def post_process(state: AssistantState) -> AssistantState:
    """
    Persist the AI turn into the graph so future turns can reference it.
    Also update conversation_history for multi-turn LLM calls.
    """
    if state["mode"] == "graph" and state.get("context_graph"):
        cg: ContextGraph = state["context_graph"]
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn = conversation_turn_node(
            turn_id=turn_id,
            role="assistant",
            content=state["ai_response"],
            intent="response",
        )
        cg.add_node(turn)
        cg.add_edge(state["user_id"], turn_id, rel="HAD_TURN")

    # Append to flat history
    history = list(state.get("conversation_history") or [])
    history.append({"role": "user", "content": state["user_message"]})
    history.append({"role": "assistant", "content": state["ai_response"]})

    return {**state, "conversation_history": history}


# ─────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────

def build_assistant_graph() -> Any:
    """
    Returns a compiled LangGraph runnable.

    Flow:
      ingest_message → update_graph → query_graph → build_prompt
                     → generate_response → post_process → END
    """
    workflow = StateGraph(AssistantState)

    workflow.add_node("ingest_message", ingest_message)
    workflow.add_node("update_graph", update_graph)
    workflow.add_node("query_graph", query_graph)
    workflow.add_node("build_prompt", build_prompt)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("post_process", post_process)

    workflow.set_entry_point("ingest_message")
    workflow.add_edge("ingest_message", "update_graph")
    workflow.add_edge("update_graph", "query_graph")
    workflow.add_edge("query_graph", "build_prompt")
    workflow.add_edge("build_prompt", "generate_response")
    workflow.add_edge("generate_response", "post_process")
    workflow.add_edge("post_process", END)

    return workflow.compile()