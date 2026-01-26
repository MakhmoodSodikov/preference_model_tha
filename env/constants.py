import os
from dotenv import load_dotenv

load_dotenv()

# Chinchilla const params
L_IRRED_CONSTANT = 1.05
E_CONSTANT = 1500
B_CONSTANT = 1500
OOM_ERROR_PCT = 0.1

EXPERIMENTS_BUDGET = 1e18
TOTAL_BUDGET = 1e23

MIN_USED_BUDGET_PCT = 0.2  # minimal percentage of the exp budget spent on experiments
MIN_DATA_LOG_RANGE = 1  # minimal expected log range (diversity) of generated datapoints
MIN_HISTORY_LENGTH = 5  # minimal number of datapoints in the training history (tool calls)
MIN_ALPHA = 0.28
MAX_ALPHA = 0.42
MIN_BETA = 0.22
MAX_BETA = 0.35

BUDGET_ERROR_TOLERANCE = 0.01  # +-1% to fluctuate around the theoretical cost constraint C = 6 * N * D
OPS_COUNT = 6.0  # number of FWD + BCKWD operations in a single layer (https://arxiv.org/pdf/2203.15556)
USE_MSE_LOSS_GAP = False
LOSS_GAP_PENALTY = 50
ALLOWED_LOG_ERROR = 1
MAX_BUDGET_REWARD_PCT = 0.1
LOSS_GAP_REWARD_WEIGHT = 0.5  # 50% wegith for right loss prediction in the total reward
SUCCESS_RATE = 0.85
MIN_TOTAL_CORE_REWARD_FOR_BONUS = 0.8
ORACLE_NOISE_VARIANCE_PCT = 0.01

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")
ANTHROPIC_API_KEY = os.environ.get("TAKE_HOME_ASSGN_PREFERENCE_MODEL_ANTHROPIC_API_KEY")
