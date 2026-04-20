"""
Tool-Calling Agent using the OpenAI Agents SDK.

This module implements a multi-turn conversational agent that uses the OpenAI
Agents SDK to handle tool calls. The agent receives a task from the environment,
invokes tools as needed, and responds to the user over multiple turns.
"""
import json
from typing import Any, Callable, Dict, List, Optional

from agents import Agent as OpenAIAgent, Runner
from agents.tool import FunctionTool
import litellm

from src.agents.base import Agent
from src.envs.base import Env
from src.types import Action, RESPOND_ACTION_NAME, SolveResult


def tools_info_to_oai_tools(
    tools_info: List[Dict[str, Any]],
    env: Env,
    done_callback: Callable[[float], None],
) -> List[FunctionTool]:
    oai_tools = []
    for tool_info in tools_info:
        func_info = tool_info["function"]
        tool_name = func_info["name"]
        description = func_info.get("description", "")
        parameters = func_info.get("parameters", {})

        def make_handler(name):
            async def on_invoke_tool(ctx, args_json: str) -> str:
                action_response = ""
                ############################################################
                # STUDENT IMPLEMENTATION START
                # 1. Construct an Action with the tool name and kwargs, and execute it via env.step().
                # 2. If the environment signals completion, invoke done_callback with the reward.
                # 3. Store the observation string in action_response.
                kwargs = json.loads(args_json)
                env_response = env.step(Action(name=name, kwargs=kwargs))
                if env_response.done:
                    done_callback(env_response.reward)
                action_response = env_response.observation
                ############################################################
                # STUDENT IMPLEMENTATION END
                ############################################################
                return action_response
            
            return on_invoke_tool

        oai_tools.append(FunctionTool(
            name=tool_name,
            description=description,
            params_json_schema=parameters,
            on_invoke_tool=make_handler(tool_name),
        ))
    return oai_tools


class ToolCallingAgentOpenAI(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        pricing = litellm.model_cost.get(self.model, {})
        input_rate = pricing.get("input_cost_per_token", 0.0)
        output_rate = pricing.get("output_cost_per_token", 0.0)


        state = {"reward": 0.0, "done": False}

        def done_callback(reward: float):
            state["done"] = True
            state["reward"] = reward

        openai_agent_tools = tools_info_to_oai_tools(self.tools_info, env, done_callback)
        openai_agent = None
        ############################################################
        # STUDENT IMPLEMENTATION START
        # Instantiate an OpenAIAgent and assign it to openai_agent.
        openai_agent = OpenAIAgent(
            name="retail-agent",
            instructions=self.wiki,
            model=self.model,
            tools=openai_agent_tools,
        )
        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################
        if not openai_agent:
            raise NotImplementedError("[TODO] Initialize openai_agent with OpenAIAgent!")

        input_messages = [{"role": "user", "content": obs}]  # initial user's request
        for _ in range(max_num_steps):
            if state["done"]:
                break
            
            agent_text = None
            ############################################################
            # STUDENT IMPLEMENTATION START
            # 1. Run openai_agent with Runner.run_sync() on input_messages.
            # 2. Save the agent's final natural language response to agent_text.
            # 3. Update total_cost.
            result = Runner.run_sync(openai_agent, input_messages)
            agent_text = result.final_output
            for raw in result.raw_responses:
                total_cost += (
                    raw.usage.input_tokens * input_rate
                    + raw.usage.output_tokens * output_rate
                )
            ############################################################
            # STUDENT IMPLEMENTATION END
            ############################################################
            if not agent_text:
                raise NotImplementedError("[TODO] Generate agent_text from OpenAI Agent SDK Runner class! ")
            if total_cost == 0:
                raise ValueError("Make sure to update total_cost!")
            
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_text})
            env_response = env.step(action)
            state["reward"] = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if env_response.done:
                state["done"] = True
                break

            # Append user's reply to history for next turn
            input_messages = result.to_input_list() + [
                {"role": "user", "content": env_response.observation}
            ]

        if not state["done"]:
            reward_result = env.calculate_reward()
            state["reward"] = reward_result.reward

        messages = []
        for item in result.to_input_list():
            if isinstance(item, dict):
                messages.append(item)
            else:
                messages.append({"role": "unknown", "content": str(item)})

        return SolveResult(
            reward=state["reward"],
            messages=messages,
            info=info,
            total_cost=total_cost,
        )
