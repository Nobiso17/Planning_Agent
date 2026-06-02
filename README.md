# Task_Agent

A minimal but production-ready agentic workflow built with **LangGraph** and **Claude**.

## Architecture

```
User Goal
    │
    ▼
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ Planner │────▶│ Executor │────▶│ Executor │────▶│Summarize│────▶ Report
│         │     │ (task 1) │     │ (task 2) │ ... │         │
└─────────┘     └──────────┘     └──────────┘     └─────────┘
                      │
                      ▼
                   Tools
              (search, calc, …)
```

| Node | What it does |
|------|-------------|
| **Planner** | Asks Claude to decompose the goal into 3-5 sub-tasks (returns JSON) |
| **Executor** | Runs one sub-task at a time; may call a tool via `TOOL:<name>:<input>` |
| **Summarizer** | Combines all task results into a final report |

Routing: after each executor run, a conditional edge checks whether more tasks remain and loops back, or moves on to the summarizer.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run with the built-in demo goal
python main.py

# 4. Or pass your own goal
python main.py "Analyse the pros and cons of microservices vs monoliths and write a decision checklist"
```

## Adding tools

Edit `tools.py`:

```python
# 1. Write your function
def _my_tool(input_str: str) -> str:
    return "result"

# 2. Register it
TOOLS.append({"name": "my_tool", "description": "What it does"})
_REGISTRY["my_tool"] = _my_tool
```

The executor will automatically discover it and tell the LLM it exists.

## Swapping in a real web search

Replace the stub in `tools.py → _web_search` with any search API:

```python
# Example: Tavily
from tavily import TavilyClient
client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def _web_search(query: str) -> str:
    results = client.search(query, max_results=3)
    return "\n".join(r["content"] for r in results["results"])
```

## File structure

```
task_agent/
├── agent.py        # LangGraph graph, nodes, routing logic
├── tools.py        # Tool definitions + implementations
├── rag.py          # RAG module 
├── main.py         # CLI entrypoint
├── requirements.txt
└── README.md
```
