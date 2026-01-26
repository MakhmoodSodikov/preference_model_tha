import random

import numpy as np
from scipy.optimize import minimize
from .logger import get_logger
from .constants import *
from .agent_behaviour_analyzer import AgentBehaviourAnalyzer
import io
import sys

logger = get_logger(__name__)


class ScalingLawEnv:
	def __init__(self):
		self.alpha = np.random.uniform(MIN_ALPHA, MAX_ALPHA)
		self.beta = np.random.uniform(MIN_BETA, MAX_BETA)
		self.E = E_CONSTANT
		self.B = B_CONSTANT
		self.L_irred = L_IRRED_CONSTANT

		self.exp_budget = EXPERIMENTS_BUDGET
		self.used_budget = 0
		self.total_budget = TOTAL_BUDGET

		self.n_true, self.d_true, self.l_min_true = self.analytical_min_loss()

		self.agent_behaviour_analyzer = AgentBehaviourAnalyzer()

		logger.setup(f"--- The World Initialized! ---")
		logger.setup(f"--- Hidden params: alpha={self.alpha:.4f}, beta={self.beta:.4f} ---\n")

	def iterative_min_loss(self):
		"""Used only for debugging and testing purposes, however,
		   the analytical solution is better, see exp/check_loss_formula.py"""

		def objective(n):
			d = self.total_budget / (OPS_COUNT * n)
			return (self.E / (n ** self.alpha)) + (self.B / (d ** self.beta)) + self.L_irred

		res = minimize(objective, x0=1e10, bounds=[(1e7, 1e14)])
		self.n_true = res.x[0]
		self.d_true = self.total_budget / (OPS_COUNT * self.n_true)
		self.l_min_true = res.fun

	def analytical_min_loss(self):
		"""Based on the Scaling Law formula, see Chinchilla paper (https://arxiv.org/pdf/2203.15556)"""
		g = (self.alpha * self.E) / (self.beta * self.B)

		exponent = 1.0 / (self.alpha + self.beta)
		compute_factor = (self.total_budget / OPS_COUNT) ** self.beta

		n_true = (g * compute_factor) ** exponent
		d_true = self.total_budget / (OPS_COUNT * n_true)
		l_min_true = self.get_loss(n_true, d_true)

		return n_true, d_true, l_min_true

	def training_oracle(self, n_params, d_tokens):
		"""This function imitates the training oracle of the agent, given the
		   number of parameters and the number of tokens. It returns the loss
		   value and the cost of the computation and the remaining budget.

		   Also, to simulate real-life training, we add some noise to the loss as like as OOM/CUDA error simlation."""

		if random.random() < OOM_ERROR_PCT:
			logger.env("Oracle raised and OOM error, passing it to agent.")
			return {"description": "Model returned undefined error in CUDA. Rerun your test.",
			        "remaining_budget": float(self.exp_budget - self.used_budget)}

		cost = OPS_COUNT * n_params * d_tokens

		logger.env(f"Training oracle called with N = {n_params} and D = {d_tokens} and C = {cost}")

		self.agent_behaviour_analyzer.add_oracle_output(n_params, d_tokens)

		# check that agent called the oracle at least once
		if not self.agent_behaviour_analyzer.training_oracle_flag:
			self.agent_behaviour_analyzer.training_oracle_flag = True

		if self.used_budget + cost > self.exp_budget:
			return logger.warning(f"Budget exceeded by an Agent! Remaining: "
			                      f"{self.exp_budget - self.used_budget:.2e}, "f"tried to allocate: {cost:.2e}")

		self.used_budget += cost

		# compute min loss
		theoretical_loss = self.get_loss(n_params, d_tokens)

		# add some Gaussian noise to make predictions more realistic - up to 5%
		# training models in real life is not deterministic, so we add some chaotic noise
		noise = np.random.normal(0, ORACLE_NOISE_VARIANCE_PCT * theoretical_loss)
		observed_loss = theoretical_loss + noise

		return {
			"val_loss": float(observed_loss),
			"cost_flops": float(cost),
			"remaining_budget": float(self.exp_budget - self.used_budget)
		}

	def execute_python(self, code: str) -> str:
		"""This method executes arbitrary Python code and returns the output.
		TODO: add more security measures to prevent arbitrary code execution.
		TODO: add checks of the code for sabotage/alignment hacks and express them in reward.
		"""
		output = io.StringIO()

		# register usage of the tools (at least once)
		if not self.agent_behaviour_analyzer.tools_flag: self.agent_behaviour_analyzer.tools_flag = True

		try:
			# Добавляем numpy в контекст для агента
			sys.stdout = output
			exec_globals = {"np": np}
			exec(code, exec_globals)
			sys.stdout = sys.__stdout__
			return output.getvalue() or "Code executed successfully (no output)."
		except Exception as e:
			sys.stdout = sys.__stdout__
			return f"Error: {str(e)}"

	def get_available_budget(self):
		available_budget = self.exp_budget - self.used_budget
		logger.env(f"Available budget: {available_budget:.2e}")
		return available_budget

	def get_loss(self, n, d):
		return (self.E / (n ** self.alpha)) + (self.B / (d ** self.beta)) + self.L_irred

	def composite_reward(self, n_pred, d_pred):
		"""
		Calculates the total episode reward based on the agent behaviour and the agent's obtained results.

		There is more than one optimal point, but it is essential to predict the right order of the N, since
	    it is the only parameter that affects the computational complexity of the future training.

		We want our model not to only find N that minimizes the Loss but also keep N as small as possible.

		This idea is based on Iso-FLOP technique (https://arxiv.org/pdf/2203.15556) and is essential to get the cheapest
		training in the real training phase keeping theoretical loss minimal as possible.
		"""
		l_min_pred = self.get_loss(n_pred, d_pred)

		# calculate the loss discrepancy (gap) between the theoretical min loss and the predicted by the agent
		if USE_MSE_LOSS_GAP:
			loss_gap = np.square(l_min_pred - self.l_min_true)
		else:
			loss_gap = max(0, (l_min_pred - self.l_min_true) / (self.l_min_true - self.L_irred))

		loss_gap_reward = np.exp(-LOSS_GAP_PENALTY * loss_gap)

		# calculate the log-error between magnitude orders of n_pred and n_true
		log_error = abs(np.log10(n_pred) - np.log10(self.n_true))
		log_error_reward = max(0, ALLOWED_LOG_ERROR - log_error)  # 1.0 if log_error <= ALLOWED_LOG_ERROR, 0.0 otherwise

		# if the agent predicted right loss with minimal computational effort, give him +5% reward
		budget_reward = (self.exp_budget - self.used_budget) / self.exp_budget

		# total core reward - 50% weight for the loss gap (Loss minimized), 50% weight for the log error (right scaling)
		total_core_reward = (loss_gap_reward ** LOSS_GAP_REWARD_WEIGHT) * (log_error_reward ** (1 - LOSS_GAP_REWARD_WEIGHT))

		# up to 5% bonus if agent predicted right with the budget economy
		if total_core_reward > MIN_TOTAL_CORE_REWARD_FOR_BONUS:
			total_core_reward += MAX_BUDGET_REWARD_PCT * budget_reward

		# behavioural shaping reward
		# set used budget and get behavioural reward
		self.agent_behaviour_analyzer.set_used_budget(self.used_budget)
		total_behaviour_reward = self.agent_behaviour_analyzer.get_behaviour_reward(n_pred, d_pred)

		total_episode_reward = total_core_reward * total_behaviour_reward

		return {
			"total_episode_reward": round(min(total_episode_reward, 1.0), 4),
			"used_budget_pct": round(self.used_budget / self.exp_budget, 4),
			"real_loss": round(self.l_min_true, 4),
			"predicted_loss": round(l_min_pred, 4),
			"breakdown": {
				"model_quality": round(loss_gap_reward, 4),
				"scaling_understanding": round(log_error_reward, 4),
				"budget_efficiency": round(budget_reward, 4),
				"diversity": self.agent_behaviour_analyzer.exp_diversity_flag,
				"total_core_reward": round(total_core_reward, 4),
				"total_behaviour_reward": bool(total_behaviour_reward)
			},
			"behaviour": {
				"tools_flag": bool(self.agent_behaviour_analyzer.tools_flag),
				"training_oracle_flag": bool(self.agent_behaviour_analyzer.training_oracle_flag),
				"exp_diversity_flag": bool(self.agent_behaviour_analyzer.exp_diversity_flag),
				"budget_flag": bool(self.agent_behaviour_analyzer.budget_flag),
				"budget_constraint_flag": bool(self.agent_behaviour_analyzer.budget_constraint_flag),
				"history": self.agent_behaviour_analyzer.oracle_history,
			}
		}

	def evaluate_solution(self, n_pred, d_pred):
		composite_reward = self.composite_reward(n_pred, d_pred)

		return {
			"reward_breakdown": composite_reward,
			"true_n": f"{self.n_true:.2e}",
			"true_d": f"{self.d_true:.2e}",
			"pred_n": f"{n_pred:.2e}",
			"pred_d": f"{d_pred:.2e}",
			"verdict": "SUCCESS" if composite_reward["total_episode_reward"] > SUCCESS_RATE else "FAILURE"
		}
