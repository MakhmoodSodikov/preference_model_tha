# preference_model_tha

## Installation 

For testing and development purposes for this environment I used `Python 3.13.5`. 

Create new .venv using pyenv with the compatible version of Python and to install the requirements run in the main directory:

`pip install -r requirements.txt`.

Create the .env using .env.example as a template:

- You need to set the `ANTHROPIC_MODEL` variable to the name of the Anthropic model you want to use.
- You need to set the `TAKE_HOME_ASSGN_PREFERENCE_MODEL_ANTHROPIC_API_KEY` variable to your Anthropic API key.

Now, you can run the application simply using `python run.py`.

It will run one episode of with the Anthropic model you chose and with the default system prompt. 

If you want to play around with different prompts or models, you can import `run_research_agent_episode(API_KEY, system_instruction_default)` from `run.py` and run it. 

## Logging 

By default, the logs are printed to stdout using `DEBUG=False` level of verbosity. You can observe and tweak this parameter in the `env/logger.py` module.

All the logging tools are implemented using default Python `logging` module in the `env/logger.py` module.

Different levels of logging are also available: 

- `DEBUG`, `INFO`, `ERROR`- by default;
- `SETUP` - for setup steps;
- `ENV` - for environment interactions/variables status;
- `AGENT` - for agent's reasoning outputs/actions;

You can see complete logging outputs for each experiment from `experiments.ipynb` in the `logs` directory.

The name structure of the log files is as follows: `[MODEL_NAME]_test_[PROMPT_NAME].txt`.

## Environment 

### AgentBehaviourAnalyzer class 

All the environment classes and modules are located in the `env/` directory.

`agent_behaviour_analyzer.py` is the module that contains the `AgentBehaviourAnalyzer` class that is used to analyze the agent's behaviour during the episode.

It collects the needed history data about the agent's actions and reasoning outputs and stores it in a different flags and lists to be used later for agent's behaviour analysis.

Finally, it returns the final agent's behaviour score as a boolean value that later is being multiplied by the `core_reward` calculated in the `env.scaling_law_env.ScalingLawEnv` class.

#### Data fields of the `AgentBehaviourAnalyzer` class

This class contains the following data fields:

- `tools_flag: bool = False` - flag that indicates whether the agent used any tools during the episode;
- `training_oracle_flag: bool = False` - flag that indicates whether the agent used the training oracle during the episode;
- `exp_diversity_flag: bool = False` - flag that indicates whether the agent used the experiment diversity oracle during the episode: experiments both diverse (more than 5 points) and sparse (at least 1 magnitude order)
- `budget_flag: bool = True` - flag that indicates whether the agent used the good amount of the budget during the episode (>=20% and <=100%);
- `budget_constraint_flag: bool = False` - flag that indicates whether the agent's final prediction is within the budget constraints (6*N*D);
- `exp_budget: float = EXPERIMENTS_BUDGET` - the budget allowed for experiments;
- `used_budget: float = -1.0` - the percentage of budget used during the episode;
- `total_budget = TOTAL_BUDGET` - the total budget available for the episode;
- `oracle_history: List[Tuple[float, float]] = []` - the history of the agent's N and D values used to calculate the oracle reward;

### def recalculate_params(self, n_pred: float, d_pred: float) method

This function is used to calculate and set all the flags at the end of the episode. 

All the rules to calculate parts of the behaviour reward are implemented inside this function: 

- Used budget is less than 20% or more than 100% -> multiplier=0 (if yes), else 1
- Diversity (variance) of the experiments' datapoints number and their values - maximal difference of magnitude orders of tested Ns and Ds is more than 1 (e.g. agent tested with N=10^15 and N=10^16 instead of collecting dense datapoints within one order) -> multiplier=1 (if yes), else 0
- Diverse experiments (collected more than 5 different points (N,D)) - multiplier=1 (if yes), else 0
- Check that the 6*N*D < 1.01*total_budget constraint is satisfied -> multiplier=1 (if yes), else 0. We give a chance to fluctuate for 1% around this value - the agent still can get 1.

We allow the agent to make a small error in the budget (up to 1%), as well as in the loss and order values, so as not to penalize too much, especially during the initial stages of agent training through policy, because it is still possible to achieve relatively small loss close to the optimal minima, but with slightly different values of N and D due to flat, almost horizontal shape of the loss around the minimum. We also give the agent a bonus if they make a small error but still save a significant amount of the exploration budget. This creates an even more accurate strategy for solving this problem, allowing the agent to fully explore the world.

### def get_behaviour_reward(self, n_pred: float, d_pred: float) -> float method

This method aggregates all the flags and reward coefficients and calculates the final behaviour reward.

The formula is simple as follows: 

`(self.budget_flag * self.exp_diversity_flag * self.training_oracle_flag * self.tools_flag * self.budget_constraint_flag)`  

New flags can be implemented and added to this formula as needed.

### ScalingLawEnv class

`ScalingLawEnv` is the main class that implements the environment. Currently, it is not the subclass of any frameworks (e.g. `gym.Env`), but it simply could be transitioned to the required notation if needed, since it follows almost the same patterns and architecture design rules. 

It is used to create the environment for the agent to interact with.

### Data fields of the `ScalingLawEnv` class

- `alpha = np.random.uniform(MIN_ALPHA, MAX_ALPHA)` - the Alpha parameter of the scaling law;
- `beta = np.random.uniform(MIN_BETA, MAX_BETA)` - the Beta parameter of the scaling law;
- `E = E_CONSTANT` - the E parameter of the scaling law;
- `B = B_CONSTANT` - the B parameter of the scaling law;
- `L_irred = L_IRRED_CONSTANT` - the irreducible part of the loss according to the Chinchilla scaling law;
- `exp_budget = EXPERIMENTS_BUDGET` - the budget allowed for experiments;
- `used_budget = 0` - the percentage of budget used during the episode;
- `total_budget = TOTAL_BUDGET` - the total budget available for the episode;
- `n_true, self.d_true, self.l_min_true = self.analytical_min_loss()` - the optimal (true) values of N and D calculated for the current setup of the environment;
- `agent_behaviour_analyzer = AgentBehaviourAnalyzer()` - the instance of the `AgentBehaviourAnalyzer` class that is used to analyze the agent's behaviour during the episode;

### def training_oracle(self, n_params, d_tokens) method 

This is the main function that is called by the agent to calculate the oracle reward. It takes the predicted values of N and D and calculates the loss according to the Chinchilla scaling law. 

At the same time, it also updates the `used_budget` field of the `ScalingLawEnv` class and passes needed N and D values to the `agent_behaviour_analyzer`.

It also raises the error with `env.constants.OOM_ERROR_PCT` probability to simulate the out-of-memory error that might occur in real world experiments.

We also add some randomness to the loss calculation to make the experiments more realistic: 

```python
noise = np.random.normal(0, ORACLE_NOISE_VARIANCE_PCT * theoretical_loss)
observed_loss = theoretical_loss + noise
```

It is simply a Gaussian noise with a standard deviation equal to the percentage of the theoretical loss that is added to the loss (by default, `ORACLE_NOISE_VARIANCE_PCT = 0.05`).

### def execute_python(self, code: str) -> str method

This method executes arbitrary Python code and returns the output.

TBD: add more security measures to prevent arbitrary code execution.

TBD: add checks of the code for sabotage/alignment hacks and express them in reward.

Using `numpy` package is allowed for the agent. 

### def get_available_budget(self) method 

Getted for the available budget for the agent.

### def get_loss(self, n, d) method 

Calculates the loss according to the Chinchilla scaling law.

### def composite_reward(self, n_pred, d_pred) method 

Calculates the final reward for the agent based on the behaviour reward and the oracle reward.

Core reward part: 

- Judge calculates the loss_gap between L_min_true and L_min_pred: loss_gap = max(0, (l_min_pred - self.l_min_true) / (self.l_min_true - self.L_irred). Final reward component for correct loss gap is loss_gap_reward = np.exp(-LOSS_GAP_PENALTY * loss_gap)
- Calculate the log-error between magnitude orders of n_pred and n_true: log_error = abs(np.log10(n_pred) - np.log10(self.n_true)); log_error_reward = max(0, ALLOWED_LOG_ERROR - log_error)
- Total core reward is loss_gap_reward**(0.5)*log_error_reward**(0.5). I decided to use the exponent reward averaging (weighted product) due to its good properties in further training using PPO or any other RL algo, see https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1808.pdf for reference
- If the agent achieves total core reward > 0.8, give him a bonus up to 5% if he used less computational efforts to achieve this result (e.g. 0.05*(exp_budget-total_budget)/total_budget)

Weights for each component of the core reward might be changed tweaking the `env.constants.LOSS_GAP_REWARD_WEIGHT`

! TBD: However, I realized, that it is possible to improve this core reward by making both parts consistent, for example: 

```python
log_error = abs(np.log10(n_pred) - np.log10(self.n_true))
log_error_reward = r_n = np.exp(-LOG_ERROR_PENALTY_COEFF * log_error)
```

It should make the training process smoother and more stable for the agent.

Returns the composite reward dictionary with all the components of the reward: 

```python
{
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
```

### def evaluate_solution(self, n_pred, d_pred) method 

Main interface to be called from outside to evaluate the agent's solution. It calculates the final reward for the agent and returns it in a dictionary with all comprehensive information about the agent's behaviour and reward breakdown.

```python
{
	"reward_breakdown": composite_reward,
	"true_n": f"{self.n_true:.2e}",
	"true_d": f"{self.d_true:.2e}",
	"pred_n": f"{n_pred:.2e}",
	"pred_d": f"{d_pred:.2e}",
	"verdict": "SUCCESS" if composite_reward["total_episode_reward"] > SUCCESS_RATE else "FAILURE"
}
```

### def analytical_min_loss(self) method 

Calculates the minimum theoretical loss for the current setup of the environment based on the Scaling Law formula, see Chinchilla paper (ref. https://arxiv.org/pdf/2203.15556)


## Miscellaneous 

- The `logger.py` module contains the `Logger` class that is used to log all the important events during the experiment.
- The `constants.py` module contains all the constants used in the project.
- The `prompts.py` module contains all the prompts used in the project.
- The `exp/` directory contains the all auxiliary scripts used in debugging and experimentation stages. 
- The `experiments.ipynb` notebook contains main experiments and analysis of the results.
- The `requirements.txt` file contains all the required dependencies for the project.
- The `.env.example` file contains the example of the `.env` file that should be created in the root directory of the project.
- The `run.py` file contains the main entry point of the project.


## Experiments results 

**NOTE**: You can observe all the results of the experiments in the `logs` directory and in the `experiments.ipynb` notebook with comprehensive comments from me. 

Here you see the aggregated results of the experiments which are presented in the tables below. 

### Metrics Breakdown

Some results of the episodes are presented in the table below:

| model_name   | prompt                         | verdict   |   pred_d |   pred_n |   true_d |   true_n |   budget_efficiency | diversity   |   model_quality |   pred_loss |   real_loss | budget_used %   |
|:-------------|:-------------------------------|:----------|---------:|---------:|---------:|---------:|--------------------:|:------------|----------------:|------------:|------------:|:----------------|
| haiku-4-5    | default_prompt                 | FAILURE   | 1.83e+11 | 9.13e+10 | 9.37e+12 | 1.78e+09 |              0.0554 | False       |          0      |      3.6097 |      2.6077 | 94.46%          |
| sonnet-4-5   | default_prompt                 | FAILURE   | 1.1e+13  | 1.51e+09 | 4.16e+11 | 4.01e+10 |              0.0053 | True        |          0      |      1.6725 |      1.4364 | 99.47%          |
| sonnet-4-5   | prompt_with_hints              | FAILURE   | 1.26e+14 | 1.32e+08 | 6.04e+13 | 2.76e+08 |              0.2611 | True        |          0.2914 |      3.1912 |      3.1397 | 73.89%          |
| sonnet-4-5   | prompt_with_hints_and_guidance | SUCCESS   | 4.83e+12 | 3.45e+09 | 4.8e+12  | 3.47e+09 |              0.1941 | True        |          0.9999 |      2.5578 |      2.5578 | 80.59%          |

Results of the experiments with exact models and prompts

### Haiku-4-5

First run, default_prompt:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.018  |
|       2 | haiku-4-5 | default_prompt |                 0      |
|       3 | haiku-4-5 | default_prompt |                 0      |
|       4 | haiku-4-5 | default_prompt |                 0      |
|       5 | haiku-4-5 | default_prompt |                 0      |
|       6 | haiku-4-5 | default_prompt |                 0.0277 |
|       7 | haiku-4-5 | default_prompt |                 0      |
|       8 | haiku-4-5 | default_prompt |                 0      |
|       9 | haiku-4-5 | default_prompt |                 0.0086 |
|      10 | haiku-4-5 | default_prompt |                 0.0166 |

Mean reward: 0.0071

Second run, prompt_with_hints:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.4305 |
|       2 | haiku-4-5 | default_prompt |                 0      |
|       3 | haiku-4-5 | default_prompt |                 0      |
|       4 | haiku-4-5 | default_prompt |                 0      |
|       5 | haiku-4-5 | default_prompt |                 0.1847 |
|       6 | haiku-4-5 | default_prompt |                 0      |
|       7 | haiku-4-5 | default_prompt |                 0.8497 |
|       8 | haiku-4-5 | default_prompt |                 0.0729 |
|       9 | haiku-4-5 | default_prompt |                 0.3158 |
|      10 | haiku-4-5 | default_prompt |                 0      |

Mean reward: 0.1854

Third run, prompt_with_hints_and_guidance:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.4379 |
|       2 | haiku-4-5 | default_prompt |                 0.8195 |
|       3 | haiku-4-5 | default_prompt |                 0      |
|       4 | haiku-4-5 | default_prompt |                 0.0204 |
|       5 | haiku-4-5 | default_prompt |                 0      |
|       6 | haiku-4-5 | default_prompt |                 0      |
|       7 | haiku-4-5 | default_prompt |                 0      |
|       8 | haiku-4-5 | default_prompt |                 0.1172 |
|       9 | haiku-4-5 | default_prompt |                 0.1837 |
|      10 | haiku-4-5 | default_prompt |                 0.2367 |

### Sonnet-4-5

First run, default_prompt:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.0993 |
|       2 | haiku-4-5 | default_prompt |                 0.0088 |
|       3 | haiku-4-5 | default_prompt |                 0.4446 |
|       4 | haiku-4-5 | default_prompt |                 0.026  |
|       5 | haiku-4-5 | default_prompt |                 0.0949 |
|       6 | haiku-4-5 | default_prompt |                 0      |
|       7 | haiku-4-5 | default_prompt |                 0.5984 |
|       8 | haiku-4-5 | default_prompt |                 0.0806 |
|       9 | haiku-4-5 | default_prompt |                 0.5781 |
|      10 | haiku-4-5 | default_prompt |                 0.7332 |

Mean reward: 0.2664

Second run, prompt_with_hints:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.5917 |
|       2 | haiku-4-5 | default_prompt |                 0.903  |
|       3 | haiku-4-5 | default_prompt |                 0.1143 |
|       4 | haiku-4-5 | default_prompt |                 0.9705 |
|       5 | haiku-4-5 | default_prompt |                 0.445  |
|       6 | haiku-4-5 | default_prompt |                 0.9814 |
|       7 | haiku-4-5 | default_prompt |                 0.5726 |
|       8 | haiku-4-5 | default_prompt |                 0.6836 |
|       9 | haiku-4-5 | default_prompt |                 0.9793 |
|      10 | haiku-4-5 | default_prompt |                 1      |

Mean reward: 0.7241

Third run, prompt_with_hints_and_guidance:

|   Run # | Model     | Prompt         |   Total Episode Reward |
|--------:|:----------|:---------------|-----------------------:|
|       1 | haiku-4-5 | default_prompt |                 0.5461 |
|       2 | haiku-4-5 | default_prompt |                 0.9086 |
|       3 | haiku-4-5 | default_prompt |                 0.994  |
|       4 | haiku-4-5 | default_prompt |                 0.9603 |
|       5 | haiku-4-5 | default_prompt |                 1      |
|       6 | haiku-4-5 | default_prompt |                 0.9889 |
|       7 | haiku-4-5 | default_prompt |                 0.75   |
|       8 | haiku-4-5 | default_prompt |                 0.984  |
|       9 | haiku-4-5 | default_prompt |                 0.7246 |
|      10 | haiku-4-5 | default_prompt |                 1      |

Mean reward: 0.8857

### Final thoughts

We see that agent outperformed the task with 70% success rate, when we almost gave him the full solution and step-by-step guidance in the prompt. Even though the model is not perfect, so it's possible to fine-tune it further.

And at the same point, prompt appears to be close to complete guided instruction and further prompt enhancing will just lead to desired behaviour pursuing or data leakage in the prompt. It's almost impossible to optimize the prompt without adding more hints that will limit model's freedom.
