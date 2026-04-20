# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HW3: Building and evaluating retail customer-service LLM agents. Agents use tool-calling to authenticate users, look up orders/products, and execute actions (cancel, return, exchange, payment change) against a mock retail database.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
export OPENAI_API_KEY=<key>

# Part 1 — unit tests for 4 tool implementations
pytest tests/test_tools.py -v

# Part 2 — integration tests for agent implementations
pytest tests/test_agents.py -v

# Part 3 — run agent on evaluation tasks
python run.py --agent-strategy tool-calling-langchain --task-split part3
python run.py --agent-strategy tool-calling-openai --task-split part3
# --task-split options: test | part3
# --agent-strategy options: tool-calling | tool-calling-langchain | tool-calling-openai
```

## Architecture

### Core Loop

```
run.py (CLI) → src/run.py (parallel task runner)
  → Agent.solve(env, task_index, max_num_steps) → SolveResult
      → env.reset(task_index)          # returns initial user message
      → loop: LLM → tool call → env.step(Action) → observation → repeat
      → env.calculate_reward()         # score vs. ground-truth actions/outputs
  → save JSON to results/
```

### Environment & Tools (`src/envs/`)

`base.py` defines the `Env` abstract class with `reset(task_index)` and `step(action)`. `MockRetailDomainEnv` (`retail/env.py`) loads 16 tools and mock data. The simulated user on the other end is also LLM-powered; the episode ends when the agent calls `transfer_to_human_agents`.

Each tool (`src/envs/retail/tools/`) implements two static methods:
- `get_info()` → OpenAI function schema (name, description, parameters JSON Schema)
- `invoke(data, **kwargs)` → JSON string on success, `"Error: ..."` on failure

Tools are the **only** way to mutate the database. The `data` dict passed to `invoke` contains the full loaded dataset (users, orders, products) and is mutated in place.

### Agents (`src/agents/`)

All agents implement `Agent.solve()` from `base.py`.

- **`tool_calling_agent.py`** — reference implementation using LiteLLM. Manual ReAct loop: call LLM, parse structured tool call, call `env.step()`, append observation, repeat. **Do not modify.**
- **`tool_calling_agent_langchain.py`** — TODO Part 2a: LangGraph agent using `create_react_agent()`. Fill three TODO blocks: `run_tool` closure, graph initialization, graph invocation.
- **`tool_calling_agent_openai.py`** — TODO Part 2b: OpenAI Agents SDK. Fill three TODO blocks: `on_invoke_tool` handler, agent init, `Runner.run_sync()` invocation.

### Tasks & Scoring (`src/envs/retail/tasks*.py`)

A `Task` has a `user_id`, natural-language `instruction`, ground-truth `actions` (ordered list of tool calls with expected args), and `outputs` (strings that must appear in the agent's final response). Reward is computed on both dimensions independently.

### Key Types (`src/types.py`)

`Action(name, kwargs)`, `Task`, `SolveResult(reward, messages, info, total_cost)`, `EnvResponse(observation, reward, done, info)`.

## What Needs Implementing

**Part 1** — four tools in `src/envs/retail/tools/`:
- `find_user_id_by_name_zip.py` — case-insensitive name + zip lookup
- `cancel_pending_order.py` — refund PaymentRecords, credit gift cards
- `modify_pending_order_items.py` — swap items, recalculate prices
- `modify_pending_order_payment.py` — update payment method, adjust balances

**Part 2a** — `tool_calling_agent_langchain.py` (LangGraph)

**Part 2b** — `tool_calling_agent_openai.py` (OpenAI Agents SDK)

The reference agent (`tool_calling_agent.py`) and the retail policy (`src/envs/retail/wiki.md`) are the primary sources of truth for correct agent behavior.
