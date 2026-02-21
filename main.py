"""
main.py
-------
1. Seeds a realistic ContextGraph for a student named Maya.
2. Runs the same three-turn conversation in BASELINE mode (no graph).
3. Runs the same conversation in GRAPH mode.
4. Prints side-by-side results so the quality difference is observable.

Run:
    OPENAI_API_KEY=sk-... python main.py

Dependencies:
    pip install networkx langgraph langchain-openai
"""

from __future__ import annotations
import json
import textwrap

from context_graph import (
    ContextGraph,
    user_node, goal_node, course_node,
    assignment_node, screen_node, resource_node,
    conversation_turn_node,
)
from langgraph_flow import build_assistant_graph, AssistantState


# ─────────────────────────────────────────────────────────────
# 1.  Seed a realistic context graph for student "Maya"
# ─────────────────────────────────────────────────────────────

def seed_context_graph() -> tuple[ContextGraph, str]:
    cg = ContextGraph()

    # ── User ─────────────────────────────────────────────────
    user_id = "user_maya"
    cg.add_node(user_node(user_id, name="Maya Chen", role="student",
                          grade="11", school="Lincoln High"))

    # ── Goals ────────────────────────────────────────────────
    goal_cs = "goal_cs_degree"
    cg.add_node(goal_node(goal_cs,
                          title="Apply to Computer Science programs",
                          deadline="2024-11-01",
                          progress=45,
                          description="Targeting UC schools and CMU early decision"))
    cg.add_edge(user_id, goal_cs, rel="HAS_GOAL")

    goal_sat = "goal_sat"
    cg.add_node(goal_node(goal_sat,
                          title="Improve SAT Math score to 760+",
                          deadline="2024-10-05",
                          progress=60))
    cg.add_edge(user_id, goal_sat, rel="HAS_GOAL")

    # ── Courses ──────────────────────────────────────────────
    course_ap = "course_ap_cs"
    cg.add_node(course_node(course_ap, name="AP Computer Science A",
                            instructor="Mr. Patel", schedule="Mon/Wed/Fri 9am",
                            grade_earned="B+"))
    cg.add_edge(user_id, course_ap, rel="ENROLLED_IN")

    course_eng = "course_english"
    cg.add_node(course_node(course_eng, name="AP English Language",
                            instructor="Ms. Torres", schedule="Tue/Thu 10am",
                            grade_earned="A-"))
    cg.add_edge(user_id, course_eng, rel="ENROLLED_IN")

    # ── Assignments ──────────────────────────────────────────
    asgn_essay = "asgn_personal_essay"
    cg.add_node(assignment_node(
        asgn_essay,
        title="College Personal Statement Draft",
        due_date="2024-09-20",
        status="in_progress",
        instructions=(
            "Write a 650-word personal statement responding to Common App Prompt 1: "
            "'Some students have a background, identity, interest, or talent that is so "
            "meaningful they believe their application would be incomplete without it.'"
        ),
        course_id=course_eng,
    ))
    cg.add_edge(user_id, asgn_essay, rel="WORKING_ON")
    cg.add_edge(goal_cs, asgn_essay, rel="REQUIRES")
    cg.add_edge(asgn_essay, course_eng, rel="PART_OF")

    asgn_algo = "asgn_sorting_lab"
    cg.add_node(assignment_node(
        asgn_algo,
        title="Sorting Algorithms Lab",
        due_date="2024-09-18",
        status="pending",
        instructions="Implement merge sort and quicksort in Java; benchmark both on arrays of 1k, 10k, 100k integers.",
        course_id=course_ap,
    ))
    cg.add_edge(user_id, asgn_algo, rel="WORKING_ON")
    cg.add_edge(asgn_algo, course_ap, rel="PART_OF")

    # ── Current screen ───────────────────────────────────────
    screen_id = "screen_goals"
    cg.add_node(screen_node(screen_id, name="My Goals",
                            description="Dashboard showing all student goals and progress bars"))
    cg.add_edge(user_id, screen_id, rel="CURRENTLY_ON")

    # ── Resources linked to CS goal ──────────────────────────
    res1 = "res_commonapp"
    cg.add_node(resource_node(res1, title="Common App Guide 2024",
                              url="https://commonapp.org/apply/",
                              res_type="guide"))
    cg.add_edge(goal_cs, res1, rel="SUGGESTED")

    res2 = "res_cs_essay_tips"
    cg.add_node(resource_node(res2, title="How to Write a CS-Focused Personal Statement",
                              url="https://blog.collegevine.com/cs-personal-statement",
                              res_type="article"))
    cg.add_edge(goal_cs, res2, rel="SUGGESTED")

    # ── Simulated prior conversation (from an earlier session) ──
    prior_t1 = "turn_prior_1"
    cg.add_node(conversation_turn_node(prior_t1, role="user",
                                       content="How many words should my personal statement be?",
                                       intent="assignment_help",
                                       entities=[asgn_essay]))
    cg.add_edge(user_id, prior_t1, rel="HAD_TURN")
    cg.add_edge(prior_t1, asgn_essay, rel="REFERENCES")

    prior_t2 = "turn_prior_2"
    cg.add_node(conversation_turn_node(prior_t2, role="assistant",
                                       content="The Common App personal statement has a 650-word limit. You should aim to get close to that limit.",
                                       intent="response"))
    cg.add_edge(user_id, prior_t2, rel="HAD_TURN")

    return cg, user_id


# ─────────────────────────────────────────────────────────────
# 2.  Helpers for running and printing a conversation
# ─────────────────────────────────────────────────────────────

DIVIDER = "─" * 72


def run_conversation(turns: list[str], mode: str,
                     context_graph: ContextGraph | None,
                     user_id: str) -> list[dict]:
    """Run multiple turns and return list of {user, assistant} dicts."""
    graph_app = build_assistant_graph()
    state: AssistantState = {
        "user_id": user_id,
        "user_message": "",
        "mode": mode,
        "context_graph": context_graph,
        "subgraph_context": None,
        "system_prompt": "",
        "conversation_history": [],
        "ai_response": "",
        "inferred_intent": "",
        "referenced_entities": [],
    }

    results = []
    for turn_text in turns:
        state["user_message"] = turn_text
        state = graph_app.invoke(state)
        results.append({"user": turn_text, "assistant": state["ai_response"]})

    return results


def print_comparison(baseline: list[dict], graph_results: list[dict]) -> None:
    print(f"\n{'═'*72}")
    print("  BASELINE vs CONTEXT GRAPH — Side-by-Side Comparison")
    print(f"{'═'*72}\n")

    for i, (b, g) in enumerate(zip(baseline, graph_results), 1):
        print(f"Turn {i}: {b['user']}")
        print(DIVIDER)
        print("▶ BASELINE RESPONSE (no context graph):")
        print(textwrap.fill(b["assistant"], width=70, initial_indent="  ",
                            subsequent_indent="  "))
        print()
        print("▶ GRAPH-AWARE RESPONSE:")
        print(textwrap.fill(g["assistant"], width=70, initial_indent="  ",
                            subsequent_indent="  "))
        print(f"\n{DIVIDER}\n")


# ─────────────────────────────────────────────────────────────
# 3.  Graph snapshot printer
# ─────────────────────────────────────────────────────────────

def print_graph_snapshot(cg: ContextGraph) -> None:
    snap = cg.snapshot()
    print(f"\n{'═'*72}")
    print("  CONTEXT GRAPH SNAPSHOT")
    print(f"{'═'*72}")
    print(f"  Nodes ({len(snap['nodes'])}):  ", end="")
    type_counts: dict[str, int] = {}
    for n in snap["nodes"]:
        t = n.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("  " + ", ".join(f"{v}× {k}" for k, v in sorted(type_counts.items())))
    print(f"  Edges ({len(snap['edges'])}):  ", end="")
    rel_counts: dict[str, int] = {}
    for e in snap["edges"]:
        r = e.get("rel", "?")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print("  " + ", ".join(f"{v}× {r}" for r, v in sorted(rel_counts.items())))
    print()

    # Print edges as a simple adjacency list
    node_labels = {nid: (d.get("name") or d.get("title") or nid)
                   for nid, d in cg.g.nodes(data=True)}
    for src, dst, data in cg.g.edges(data=True):
        print(f"  [{node_labels[src]}] ──{data.get('rel','?')}──► [{node_labels[dst]}]")
    print()


# ─────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────

DEMO_TURNS = [
    "I'm not sure how to start my personal statement. Any ideas?",
    "What deadline should I keep in mind?",
    "Can you give me tips that are specific to a CS applicant?",
]


def main() -> None:
    # Seed the graph
    cg, user_id = seed_context_graph()
    print_graph_snapshot(cg)

    print("Running BASELINE conversation (no context graph)…")
    baseline_results = run_conversation(DEMO_TURNS, mode="baseline",
                                        context_graph=None, user_id=user_id)

    print("Running GRAPH-AWARE conversation…")
    graph_results = run_conversation(DEMO_TURNS, mode="graph",
                                     context_graph=cg, user_id=user_id)

    print_comparison(baseline_results, graph_results)

    # Print subgraph that was built for the final turn (for documentation)
    final_subgraph = cg.build_relevant_subgraph(user_id)
    print(f"{'═'*72}")
    print("  SUBGRAPH USED FOR FINAL TURN (JSON)")
    print(f"{'═'*72}")
    print(json.dumps(final_subgraph, indent=2, default=str))


if __name__ == "__main__":
    main()