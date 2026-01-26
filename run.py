from anthropic import Anthropic
from env import ScalingLawEnv
import json
from env.logger import get_logger, setup_logging
from env.constants import *
from pprint import pprint
from typing import TypedDict
from env.prompts import system_instruction_default


# --- custom logging config setup ---
setup_logging(debug=False)
logger = get_logger(__name__)

DEBUG_AGENT_CODE = False


def run_research_agent_episode(api_key: str, system_instruction: str) -> TypedDict:
	client = Anthropic(api_key=api_key)
	env = ScalingLawEnv()

	logger.info("Starting new episode...")

	# tools for calling
	tools = [
		{
			"name": "training_oracle",
			"description": "Simulate training and get real loss values for N and D.",
			"input_schema": {
				"type": "object",
				"properties": {
					"n_params": {"type": "number", "description": "Number of parameters (N)"},
					"d_tokens": {"type": "number", "description": "Number of tokens (D)"}
				},
				"required": ["n_params", "d_tokens"]
			}
		},
		{
			"name": "submit_solution",
			"description": f"Submit the final predicted N and D for the {TOTAL_BUDGET} FLOPs budget.",
			"input_schema": {
				"type": "object",
				"properties": {
					"predicted_n": {"type": "number"},
					"predicted_d": {"type": "number"},
					"report": {"type": "string", "description": "Your analysis and exponents estimation."}
				},
				"required": ["predicted_n", "predicted_d", "report"]
			}
		},
		{
			"name": "python",
			"description": "Execute Python for math/regression. Use print() to see results. stdout will be returned.",
			"input_schema": {
				"type": "object",
				"properties": {"code": {"type": "string"}},
				"required": ["code"]
			}
		},
		{
			"name": "check_available_resources",
			"description": "See how much FLOPs you have left.",
			"input_schema": {
				"type": "object",
				"properties": {},
			}
		},
	]

	messages = [{"role": "user", "content": f"The {EXPERIMENTS_BUDGET} experimentation budget is ready. Please begin "
	                                        f"your research to find the optimal N and D for {TOTAL_BUDGET} FLOPs."}]

	logger.info("Initial system message sent. Starting the chat loop...")
	logger.info(f"Model used: {ANTHROPIC_MODEL}")

	while True:
		response = client.messages.create(
			model=ANTHROPIC_MODEL,
			max_tokens=4000,
			system=system_instruction,
			messages=messages,
			tools=tools
		)

		messages.append({"role": "assistant", "content": response.content})
		tool_results = []
		final_data = None

		for block in response.content:
			if block.type == "text":
				logger.agent(f"Message: {block.text}")

			if block.type == "tool_use":
				logger.agent(f"Calling {block.name}")

				if block.name == "training_oracle":
					logger.agent(f"Calling training_oracle with N={block.input["n_params"]} and D={block.input["d_tokens"]}")
					res = env.training_oracle(block.input["n_params"], block.input["d_tokens"])
				elif block.name == "python":
					if DEBUG_AGENT_CODE:
						logger.agent(f"Calling python with code: {block.input["code"]}")
					res = {"output": env.execute_python(str(block.input["code"]))}
				elif block.name == "check_available_resources":
					logger.agent(f"Calling available resources")
					res = {"remaining_budget": env.get_available_budget()}
				elif block.name == "submit_solution":
					final_data = env.evaluate_solution(block.input['predicted_n'], block.input['predicted_d'])
					res = {"status": "received"}
				else:
					raise ValueError(f"Unknown tool: {block.name}")

				tool_results.append({
					"type": "tool_result",
					"tool_use_id": block.id,
					"content": json.dumps(res)
				})

		if final_data:
			logger.info(f"\n" + "=" * 30)
			logger.info(f"FINAL JUDGE RESULT: {final_data['verdict']}")
			pprint(final_data)
			logger.info("=" * 30)
			break

		if tool_results:
			messages.append({"role": "user", "content": tool_results})
		else:
			break

	return final_data


if __name__ == "__main__":
	API_KEY = os.environ.get("TAKE_HOME_ASSGN_PREFERENCE_MODEL_ANTHROPIC_API_KEY")

	run_research_agent_episode(API_KEY, system_instruction_default)
