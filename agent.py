"""
Task Planning & Execution Agent — built with LangGraph + Claude
================================================================
Improvements over v1:
  • RAG  — executor retrieves relevant knowledge-base context before each task
  • Reflection node — scores executor output; retries once if quality is low
  • Model tiering — cheap model for routing/reflection, capable model for work

Graph layout:
  planner → executor → reflector → (retry executor OR continue)
          → (loop back if tasks remain) → summarizer → END
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import rag
from tools import TOOLS, run_tool

# ── Initialise RAG (indexes docs/ folder on import) ───────────────────────────
rag.init("docs")

# ── Models ─────────────────────────────────────────────────────────────────────
# Use a capable model for planning, execution, and summarisation.
# Use a cheaper/faster model for the reflection quality-check.

llm = ChatAnthropic(model="claude-opus-4-5", max_tokens=2048)
llm_fast = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)   # reflection


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    goal: str
    tasks: list[dict]             # [{"id":1, "description":"...", "done":false}]
    current_task_index: int
    task_results: list[str]
    retry_count: int              # NEW: tracks retries for current task
    messages: Annotated[list, add_messages]
    final_report: str


# ── Node: Planner ──────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    """Break the goal into 3-5 concrete sub-tasks."""
    print("\nPLANNER — decomposing goal…")

    tool_names = ", ".join(t["name"] for t in TOOLS)
    rag_context = rag.retrieve(state["goal"], top_k=2)

    system_content = (
        "You are a task planner. Given a goal, decompose it into 3-5 concrete, "
        "actionable sub-tasks. You have access to these tools: "
        f"{tool_names}. "
        "Reply ONLY with a JSON array like:\n"
        '[{"id":1,"description":"..."},{"id":2,"description":"..."}]'
    )
    if rag_context:
        system_content += f"\n\nRelevant background knowledge:\n{rag_context}"

    system = SystemMessage(content=system_content)
    user = HumanMessage(content=f"Goal: {state['goal']}")

    response: AIMessage = llm.invoke([system, user])
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    tasks = json.loads(raw)
    for t in tasks:
        t["done"] = False

    print(f"   → {len(tasks)} tasks planned")
    for t in tasks:
        print(f"     {t['id']}. {t['description']}")

    return {
        "tasks": tasks,
        "current_task_index": 0,
        "task_results": [],
        "retry_count": 0,
        "messages": [system, user, response],
    }


# ── Node: Executor ─────────────────────────────────────────────────────────────

def _build_executor_system(rag_context: str) -> SystemMessage:
    base = (
        "You are an execution agent. You are given one task to complete. "
        "Use the available tools if helpful, then write a concise result summary.\n\n"
        "Available tools:\n"
        + "\n".join(
            f"  • {t['name']}: {t['description']}  →  TOOL:{t['name']}:<input>"
            for t in TOOLS
        )
        + "\n\nIf you call a tool, write exactly one line starting with 'TOOL:' "
        "and nothing else on that line. After seeing the tool result, write your "
        "final task summary starting with 'RESULT:'."
    )
    if rag_context:
        base += f"\n\n{rag_context}"
    return SystemMessage(content=base)


def executor(state: AgentState) -> dict:
    """Execute the current task, optionally calling a tool, with RAG context."""
    idx = state["current_task_index"]
    task = state["tasks"][idx]
    is_retry = state.get("retry_count", 0) > 0
    prefix = "RETRY" if is_retry else "  EXECUTOR"
    print(f"\n{prefix} — task {task['id']}: {task['description']}")

    # RAG: retrieve knowledge relevant to this specific task
    rag_context = rag.retrieve(task["description"], top_k=3)
    if rag_context:
        print(f"   📚 RAG: injecting {rag_context.count('—')} snippet(s)")

    system = _build_executor_system(rag_context)
    user_msg = HumanMessage(content=f"Task: {task['description']}")
    history = [system, user_msg]

    for _ in range(3):
        response: AIMessage = llm.invoke(history)
        text = response.content.strip()
        history.append(response)

        tool_line = next((l for l in text.splitlines() if l.startswith("TOOL:")), None)
        if tool_line:
            parts = tool_line.split(":", 2)
            if len(parts) == 3:
                _, tool_name, tool_input = parts
                tool_name = tool_name.strip()
                tool_input = tool_input.strip()
                print(f"   🔧 calling tool '{tool_name}' with: {tool_input!r}")
                tool_result = run_tool(tool_name, tool_input)
                print(f"   tool result: {tool_result}")
                history.append(HumanMessage(content=f"Tool result: {tool_result}"))
                continue

        result_line = next((l for l in text.splitlines() if l.startswith("RESULT:")), None)
        result = result_line[len("RESULT:"):].strip() if result_line else text
        print(f"   result: {result}")

        updated_tasks = [
            {**t, "done": True} if t["id"] == task["id"] else t
            for t in state["tasks"]
        ]

        return {
            "tasks": updated_tasks,
            "task_results": state["task_results"] + [
                f"Task {task['id']} — {task['description']}: {result}"
            ],
            "retry_count": 0,   # reset on success
            "messages": history[2:],
        }

    # Fallback if no RESULT after 3 rounds
    return {
        "tasks": state["tasks"],
        "task_results": state["task_results"] + [
            f"Task {task['id']} — no result produced"
        ],
        "retry_count": 0,
        "messages": history[2:],
    }


# ── Node: Reflector ────────────────────────────────────────────────────────────

def reflector(state: AgentState) -> dict:
    """
    Quality-check the most recent task result.
    Uses the fast model to score 1-5; triggers a retry if score <= 2
    and we haven't already retried this task.
    """
    last_result = state["task_results"][-1] if state["task_results"] else ""
    idx = state["current_task_index"] - 1   # executor already incremented
    task = state["tasks"][idx] if 0 <= idx < len(state["tasks"]) else {}
    task_desc = task.get("description", "unknown task")

    print(f"\n🔍  REFLECTOR — checking task {task.get('id', '?')} quality…")

    system = SystemMessage(content=(
        "You are a quality-checker for an AI agent. "
        "Given a task description and its result, rate the result quality 1-5.\n"
        "1=completely wrong or empty, 3=acceptable, 5=excellent.\n"
        "Reply with ONLY a single integer (1-5). No other text."
    ))
    user = HumanMessage(content=(
        f"Task: {task_desc}\n\nResult: {last_result}"
    ))

    response: AIMessage = llm_fast.invoke([system, user])
    raw_score = response.content.strip()

    try:
        score = int(raw_score[0])   # take first char in case of stray punctuation
    except (ValueError, IndexError):
        score = 3   # default: acceptable

    print(f"   Score: {score}/5")
    return {"retry_count": score}   # we repurpose retry_count to carry the score


# ── Node: Summarizer ───────────────────────────────────────────────────────────

def summarizer(state: AgentState) -> dict:
    """Compile all task results into a final report."""
    print("\n📊  SUMMARIZER — compiling report…")

    results_text = "\n".join(state["task_results"])
    system = SystemMessage(content=(
        "You are a report writer. Given a goal and completed task results, "
        "write a clear, concise final report (3-5 sentences) summarising what was "
        "accomplished and any key findings."
    ))
    user = HumanMessage(content=(
        f"Goal: {state['goal']}\n\nCompleted tasks:\n{results_text}"
    ))

    response: AIMessage = llm.invoke([system, user])
    report = response.content.strip()
    print(f"\n{'='*60}\n FINAL REPORT\n{'='*60}\n{report}\n{'='*60}\n")

    return {"final_report": report, "messages": [system, user, response]}


# ── Routing ────────────────────────────────────────────────────────────────────

_MAX_RETRIES = 1   # retry each task at most once


def after_executor(state: AgentState) -> str:
    """Always go to reflector after executor."""
    return "reflect"


def after_reflector(state: AgentState) -> str:
    """
    retry_count holds the reflection score (1-5).
    Score <= 2 AND haven't retried → retry executor (roll back index by 1).
    Otherwise → continue to next task or summarizer.
    """
    score = state.get("retry_count", 3)
    # Check if result was a retry score signal
    if score <= 2:
        # We'll re-run the last task: undo the index increment inside executor
        print(f"   Quality too low (score={score}), scheduling retry…")
        # Decrement index so executor re-runs the same task
        return "retry"

    # Advance to next task or wrap up
    if state["current_task_index"] < len(state["tasks"]):
        return "execute"
    return "summarize"


# ── Graph ──────────────────────────────────────────────────────────────────────

def _undo_index(state: AgentState) -> dict:
    """Helper node: rolls back current_task_index so executor re-runs the task."""
    print("    Rolling back task index for retry…")
    # Also remove the bad result from task_results and reset task.done
    bad_idx = state["current_task_index"] - 1
    updated_tasks = [
        {**t, "done": False} if i == bad_idx else t
        for i, t in enumerate(state["tasks"])
    ]
    trimmed_results = state["task_results"][:-1]   # drop the low-quality result
    return {
        "current_task_index": bad_idx,
        "tasks": updated_tasks,
        "task_results": trimmed_results,
        "retry_count": 99,   # sentinel: don't re-enter retry loop
    }


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reflector", reflector)
    graph.add_node("undo_index", _undo_index)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")

    graph.add_conditional_edges("reflector", after_reflector, {
        "retry": "undo_index",
        "execute": "executor",
        "summarize": "summarizer",
    })

    graph.add_edge("undo_index", "executor")
    graph.add_edge("summarizer", END)

    return graph.compile()


app = build_graph()


def run(goal: str) -> str:
    """Run the agent against a goal and return the final report."""
    initial_state: AgentState = {
        "goal": goal,
        "tasks": [],
        "current_task_index": 0,
        "task_results": [],
        "retry_count": 0,
        "messages": [],
        "final_report": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["final_report"]


if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Research best practices for Python project structure, "
        "create a project outline, and write a getting-started checklist."
    )
    run(goal)
