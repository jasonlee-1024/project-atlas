# HW3: Building and Evaluating Retail Customer-Service Agents

## Overview

In this assignment you will build and evaluate LLM-powered agents for retail customer-service tasks.

An *agent* is an LLM-based system that can call external tools to complete
multi-step tasks.  The agent acts as a customer-service representative: it
authenticates users, looks up their orders and products, and takes actions such
as cancelling orders, processing returns, or modifying pending orders.

The code skeleton is in `src/`.  You will fill in the masked sections marked
with `STUDENT IMPLEMENTATION START / END` comment blocks.

---

## Background: What Is a Tool-Calling Agent?

### The Agent

An agent is an LLM wrapped in a control loop that can take *actions* in the world by calling **tools**.  Instead of producing only plain text, the LLM can emit a structured "tool call" — a JSON object naming a function and its arguments — which the framework executes and feeds back as an observation.  The LLM then reasons over that observation and decides whether to call another tool or produce a final natural-language response to the user.

In this assignment the agent plays the role of a retail customer-service representative.  Its behavior is governed by a *policy* (the `wiki.md` file loaded as the system prompt) that specifies what the agent may and may not do: authenticate the caller before acting, request explicit confirmation before any database-mutating action, never fabricate information, and escalate to a human agent when the request falls outside its scope.

The agent loop runs until one of the following terminal conditions is reached:
- The user closes the conversation (sends `###STOP###`).
- The agent calls `transfer_to_human_agents`.
- The maximum number of steps is exceeded.

### What Are Tools?

Tools are the agent's only interface to the outside world.  Each tool is a Python class that exposes two static methods:

- **`get_info() -> dict`** — Returns an OpenAI-style function-schema JSON object (name, description, and parameter definitions).  The framework passes this schema to the LLM so it knows the tool exists and how to call it.
- **`invoke(data, **kwargs) -> str`** — The actual implementation.  It receives the shared in-memory database (`data`, containing `users`, `orders`, and `products` dicts) and the arguments the LLM supplied.  It returns either a JSON string on success or an `"Error: ..."` string on failure.

The LLM never touches the database directly.  It can only read or mutate state by calling tools, which enforces a clean separation between reasoning (LLM) and execution (tools).

The 16 retail tools cover the following capabilities:

| Category | Tools |
|---|---|
| User lookup | `find_user_id_by_email`, `find_user_id_by_name_zip`, `get_user_details` |
| Order & product info | `get_order_details`, `get_product_details`, `list_all_product_types` |
| Pending order mutations | `cancel_pending_order`, `modify_pending_order_items`, `modify_pending_order_payment`, `modify_pending_order_address` |
| Delivered order mutations | `return_delivered_order_items`, `exchange_delivered_order_items` |
| Utilities | `calculate`, `think`, `transfer_to_human_agents`, `modify_user_address` |

### How the User Interacts with the Agent

Conversations are turn-based.  On each turn the *user* sends a message; the *agent* either calls a tool (invisible to the user) or responds with natural language.  The cycle repeats until the task is resolved or the conversation ends.

The user's utterance is controlled by a second LLM, which plays the role of the user, guided by a task instruction (e.g., "cancel order #W1234567 because you no longer need it").  The simulated user reveals information incrementally and sends `###STOP###` once the task goal has been satisfied.  This is used for automated evaluation (Part 3).

A typical conversation trajectory looks like:

```
[User]   I want to cancel one of my orders.
[Agent]  Sure, I can help with that. Could you please provide me your email address
         or your name and zip code so I can look up your account?
[User]   My name is Jane Doe and my zip is 10001.
[Agent]  <calls find_user_id_by_name_zip(first_name="Jane", last_name="Doe", zip="10001")>
[Tool]   {"user_id": "u_jane_doe_123", ...}
[Agent]  I found your account. Which order would you like to cancel, and what is the reason?
[User]   Order #W9876543, I no longer need it.
[Agent]  <calls get_order_details(order_id="#W9876543")>
[Tool]   {"status": "pending", "total": 59.99, ...}
[Agent]  I'd like to cancel order #W9876543 (total $59.99) for the reason
         "no longer needed". Can you confirm (yes/no)?
[User]   Yes.
[Agent]  <calls cancel_pending_order(order_id="#W9876543", reason="no longer needed")>
[Tool]   {"status": "cancelled", ...}
[Agent]  Done! Your order has been cancelled and the refund will be processed
         within 5–7 business days.
[User]   ###STOP###
```

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=<assigned_openai_api_key>
```

> **Note on API key usage:**

> <assigned_openai_api_key> = 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

> The provided key supports only `gpt-4o` and `gpt-4o-mini` model.  The expected total cost for completing this assignment (Parts 1–3) is **under $1**.
>
> **Please be mindful of your usage. Do not share with other people outside the class.**  Avoid running large-scale experiments or repeated bulk evaluations with the key.
>
> You are encouraged to use **your own OpenAI API key** instead of the shared one.  When many students use the same key simultaneously (especially just before the deadline!!!), you may hit rate limits that slow or block your requests.  Using your own key avoids this and also lets you experiment with other models or providers (e.g., Anthropic Claude, Google Gemini) during development.
>
> **Important:** The autograder always evaluates your agent against OpenAI (`gpt-4o-mini`), so your implementation must be compatible with that model regardless of which key or model you use locally.

---

## File Structure

```
hw3
├── README.md                        <- this file
├── requirements.txt
├── run.py
├── tests/
│   ├── test_tools.py                <- Part 1 unit tests
│   └── test_agents.py               <- Part 2 integration tests
└── src/
    ├── agents/
    │   ├── base.py                  <- Agent interface (read-only)
    │   ├── tool_calling_agent.py    <- LiteLLM reference (read-only)
    │   ├── tool_calling_agent_langchain.py   <- TODO (Part 2a)
    │   └── tool_calling_agent_openai.py      <- TODO (Part 2b)
    └── envs/retail/
        ├── wiki.md                  <- agent policy (read-only)
        ├── tasks_part3.py           <- 10-task subset used for (Part 3)
        └── tools/
            ├── find_user_id_by_name_zip.py    <- TODO (Part 1)
            ├── cancel_pending_order.py         <- TODO (Part 1)
            ├── modify_pending_order_items.py   <- TODO (Part 1)
            └── modify_pending_order_payment.py <- TODO (Part 1)
```

---

## Part 1 — Implement Tool Functions (40 pts)

The 16 retail tools live in `src/envs/retail/tools/`.  Most are already
implemented.  You need to implement the `invoke()` method for the following 4 tools (the `get_info()` method is already provided for each):

| File | Points | Description |
|---|---|---|
| `find_user_id_by_name_zip.py` | 10 | Look up a user by first name, last name, and zip code (case-insensitive name match). |
| `cancel_pending_order.py` | 10 | Cancel a pending order, process refunds, and immediately credit gift-card payments. |
| `modify_pending_order_items.py` | 10 | Swap items in a pending order; compute the price difference and update payment history. |
| `modify_pending_order_payment.py` | 10 | Switch the payment method of a pending order; handle gift-card balance updates. |

Each `invoke()` method receives the shared `data` dict (containing `users`,
`orders`, and `products`) and the relevant parameters.  It must return a JSON
string on success or an `"Error: ..."` string on failure.

**Sanity check** — run the provided unit tests under the root directory:

```bash
pytest tests/test_tools.py -v
```

---

## Part 2 — Implement tool-calling Agents (30 pts)

You will implement the agent using two different frameworks.
Both agents must conform to the `Agent` interface defined in `src/agents/base.py` and produce a `SolveResult`.

The reference implementation is in `src/agents/tool_calling_agent.py` — read it carefully before you start.  It implements the raw ReAct loop manually using LiteLLM: it calls the LLM, checks whether the response contains a tool call, executes the tool via `env.step()`, appends the result to the message history, and loops.  Your task is to replicate the same behavior using higher-level frameworks that handle the inner tool-call loop for you.

### Framework Overview

#### LangGraph

[LangGraph](https://github.com/langchain-ai/langgraph) is a graph-based orchestration layer built on top of LangChain.  Agent behavior is modeled as a directed graph: nodes represent LLM calls or tool executions, and edges encode the control flow.  The `create_react_agent` helper (from `langgraph.prebuilt`) builds a standard ReAct loop graph automatically — you only need to supply an LLM and a list of tools.

**Key classes used in this assignment:**
- `ChatOpenAI` (`langchain_openai`) — the LLM node
- `StructuredTool` (`langchain_core.tools`) — wraps a Python function as a LangChain tool with a Pydantic input schema
- `create_react_agent` (`langgraph.prebuilt`) — assembles the ReAct graph

**Where to learn more:**
- Quickstart tutorial: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- `create_react_agent` API reference: https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent

When you call `graph.invoke(messages)`, LangGraph runs the full inner loop (LLM → tool → LLM → …) and returns only when the LLM produces a final text response.  This replaces the manual `while` loop in the reference implementation.

#### OpenAI Agents SDK

The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (pip package `openai-agents`, import as `agents`) is a lightweight framework from OpenAI for building tool-calling agents.  An `Agent` object bundles the model name, a system-prompt (`instructions`), and a list of `FunctionTool` objects.  `Runner.run_sync()` executes the agent on a list of messages and handles all intermediate tool calls internally, returning a `RunResult` whose `.final_output` attribute is the agent's final text response.

**Key classes used in this assignment:**
- `agents.Agent` — holds model, instructions, and tools
- `agents.Runner` — executes the agent loop
- `agents.tool.FunctionTool` — wraps an async Python handler as an OpenAI tool; the `on_invoke_tool` callback receives the raw JSON arguments string and returns the tool result string

**Where to learn more:**
- Official documentation: https://openai.github.io/openai-agents-python/
- GitHub examples: https://github.com/openai/openai-agents-python/tree/main/examples

The `on_invoke_tool` callback plays the same role as the `tool` role message in the raw API loop of the reference implementation: it receives the LLM's tool-call arguments and returns the observation that gets fed back to the model.

---

### Part 2a — LangGraph agent (15 pts)

File: `src/agents/tool_calling_agent_langchain.py`

Fill in the three `STUDENT IMPLEMENTATION` blocks:

1. **`run_tool` closure** (inside `tools_info_to_langchain_tools`) — construct an `Action`, call `env.step()`, invoke `done_callback` if the episode is done, and return the action response.
2. **Graph initialization** (inside `solve`) — instantiate a `ChatOpenAI` LLM and create a agent graph.
3. **Graph invocation** (inside the outer loop) — call `graph.invoke()` with the current message history and extract the agent's final text response.

Key classes: `ChatOpenAI`, `StructuredTool`, `create_react_agent`.

### Part 2b — OpenAI Agents SDK agent (15 pts)

File: `src/agents/tool_calling_agent_openai.py`

Fill in the three `STUDENT IMPLEMENTATION` blocks:

1. **`on_invoke_tool` handler** (inside `tools_info_to_oai_tools`) — construct an `Action`, call `env.step()`, invoke `done_callback` if the episode is done, and return the action response.
2. **Agent initialization** (inside `solve`) — instantiate an `OpenAIAgent` with the wiki as instructions and `openai_agent_tools` as its tool list.
3. **Runner invocation** (inside the outer loop) — run `Runner.run_sync()` on the current message history, save the final output to `agent_text`, and accumulate token costs using `input_rate` and `output_rate`.

Key classes: `agents.Agent`, `agents.Runner`, `agents.tool.FunctionTool`.

**Sanity check** — run the integration tests (requires `OPENAI_API_KEY`):

```bash
pytest tests/test_agents.py -v
```

---

## Part 3 — Benchmark Evaluation and Error Analysis (30 pts)

After completing Parts 1 and 2, **choose one of agents** from Part 2 to run the curated 10-task.

### Run evaluation
Run one of the command below:
```bash
# LangGraph agent
python run.py --agent-strategy tool-calling-langchain --task-split part3
```

```bash
# OpenAI Agents SDK agent
python run.py --agent-strategy tool-calling-openai --task-split part3
```

### Write-up questions

1. (10 pts) Report the Pass^1 score (The percentage of tasks completed successfully on the first attempt, the score should be printed out in the console.) for the selected agent on the 10-task subset. Save the results log into "results" folder.
2. (10 pts) What are the types of different agent failures? Pick **3 failure cases** from your results. For each one, examine the conversation trajectory and explain why the agent failed (e.g., wrong tool called, incorrect argument, partial resolution, etc.).
3. (10 pts) Any suggestion to solve the above identified issues?

---

## Bonus — Open-Ended Agent Project (20 pts)

1. Build and evaluate **any** interactive agent of your choice. You are free to pick the application domain, the agent architecture, and the framework — this is intentionally open-ended.

    **Example frameworks** (not exhaustive): Claude Anthropic SDK, LangChain/LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, raw LLM API calls, or any combination.

    **Example agent types**: research agent, shopping assistant, IT support agent, tutor agent, travel planning agent, etc.

2. Evaluation with Arksim

    [`arksim`](https://github.com/arklexai/arksim) is an open-source agent simulation and evaluation framework.  It spins up a simulated user (powered by an LLM) that converses with your agent turn-by-turn, then scores the interaction against a set of evaluation criteria.  This lets you run automated, reproducible benchmarks without needing real human testers.

    Key concepts:
    - **Simulation**: `arksim` drives a multi-turn conversation between a simulated user and your agent, following a task description you define.
    - **Evaluation**: after the conversation ends, the framework grades the agent's performance (e.g., task completion, policy adherence, response quality).

    ```bash
    pip install arksim
    ```

    Follow the [arksim documentation](https://github.com/arklexai/arksim) to run a simulation and produce results. Save all output to the `simulate_results/` folder. It can either be a screenshot from `arksim ui` or the results produced via the command line.

### Submission questions

1. **(10 pts)** Describe your agent: what task does it solve, what framework did you use, and what design choices did you make? Save the agent code in the `part4_agent/` folder and include it in your submission.
2. **(5 pts)** Save your `arksim` simulation results to the `part4_simulation/` folder and include it in your submission.
3. **(5 pts)** Reflect on your experience using the `arksim` package.  What worked well? What was confusing or missing? If you were to improve it, what would you change?

---

## Submission

Submit your completed assignment to **Gradescope** as a single zip file named `hw3.zip` with the following structure:

```
hw3.zip
├── writeup.pdf
├── src/
├── tests/
├── results/              # Part 3 evaluation results
├── bonus_agent/          # Bonus agent code
└── bonus_simulation/     # Bonus simulation results
```

> **Do NOT include your virtual environment folder.**

> After unzipping, the top-level directory should contain: `writeup.pdf`, `src/`, `tests/`, `results/`, `bonus_agent/`, and `bonus_simulation/`.

---

### writeup.pdf

Any formatting is acceptable, but the PDF must have clearly labeled sections for **Part 3** and **Part 4**.

**Part 3:**
1. State which agent you chose and its Pass^1 score.
2. Select **3 failure cases** from your results. For each case, examine the conversation trajectory and explain why the agent failed (e.g., wrong tool called, incorrect argument, partial resolution, etc.).
3. Suggest a potential fix or improvement for each of the identified failure modes.

**Bonus:**
1. Describe your agent — what task does it solve, what framework did you use, and what design choices did you make?
2. Describe the failure cases you observed from the simulation results. 
3. Reflect on your experience using the `arksim` package.  What worked well? What was confusing or missing? If you were to improve it, what would you change?


