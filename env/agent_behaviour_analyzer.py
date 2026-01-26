from .constants import *
import numpy as np
from typing import List, Tuple


class AgentBehaviourAnalyzer:
	def __init__(self):
		self.tools_flag: bool = False
		self.training_oracle_flag: bool = False
		self.exp_diversity_flag: bool = False
		self.budget_flag: bool = True
		self.budget_constraint_flag: bool = False

		self.exp_budget: float = EXPERIMENTS_BUDGET
		self.used_budget: float = -1.0  # initially set to -1.0 to check that it was ever updated
		self.total_budget = TOTAL_BUDGET

		self.oracle_history: List[Tuple[float, float]] = []

	def add_oracle_output(self, n_pred: float, d_pred: float):
		data = (n_pred, d_pred)
		self.oracle_history.append(data)

	def set_used_budget(self, budget: float):
		self.used_budget = budget

	def recalculate_params(self, n_pred: float, d_pred: float):
		# budget spending behavior shaping
		used_budget_pct = self.used_budget / self.exp_budget

		if used_budget_pct < MIN_USED_BUDGET_PCT or used_budget_pct > 1.0:  # either guessing or budget exhausted
			self.budget_flag = False

		# diversity (variance) of the experiments' datapoints number and their values
		powers = np.floor(np.log10(np.array(self.oracle_history)))  # get the max diffs in the log scale of oracle outs
		n_range, d_range = np.ptp(powers, axis=0)  # calculate the range of the max diffs

		# experiments both diverse (more than 5 points) and sparse (at least 1 magnitude order)
		# if yes, multiplier is 1.0, otherwise 0.0
		self.exp_diversity_flag = ((n_range >= MIN_DATA_LOG_RANGE and d_range >= MIN_DATA_LOG_RANGE)
		                           and (len(self.oracle_history) > MIN_HISTORY_LENGTH))

		# check that the 6*N*D < total_budget constraint is satisfied
		budget_error_pct = abs(OPS_COUNT * n_pred * d_pred - self.total_budget) / self.total_budget
		self.budget_constraint_flag = budget_error_pct <= BUDGET_ERROR_TOLERANCE

	def get_behaviour_reward(self, n_pred: float, d_pred: float) -> float:
		assert self.used_budget >= 0, \
			"`used_budget` variable is unset, please set it before calling `get_behaviour_reward()`"

		self.recalculate_params(n_pred, d_pred)

		return (self.budget_flag * self.exp_diversity_flag * self.training_oracle_flag * self.tools_flag
		        * self.budget_constraint_flag)
