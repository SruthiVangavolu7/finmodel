"""Central configuration for the ML equity signal generator."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Universe and date range
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Data handling
CACHE_MAX_AGE_DAYS = 1
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DOWNLOAD_MAX_RETRIES = 4
DOWNLOAD_BACKOFF_SECONDS = 3.0
DOWNLOAD_REQUEST_TIMEOUT = 30
DOWNLOAD_PAUSE_SECONDS = 0.5

# Feature engineering
FEATURE_WINDOWS = [5, 10, 20, 50]
RSI_WINDOWS = [7, 14]
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULTIPLIER = 2.0
VOLUME_WINDOW = 20
LAG_RETURNS = [1, 2, 3, 4, 5]
LOOKAHEAD_CORR_THRESHOLD = 0.8

# Labels
LABEL_HORIZON = 1
LABEL_THRESHOLD = 0.0

# Model and CV
MODEL_TYPE = "random_forest"  # options: "xgboost", "random_forest", "logistic"
N_SPLITS = 5
TRAIN_WINDOW = 504
TEST_WINDOW = 63
CLASSIFICATION_THRESHOLD = 0.5

# Model hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 20,
}
LOGISTIC_PARAMS = {
    "C": 0.1,
    "max_iter": 1000,
    "solver": "lbfgs",
}

# Backtest
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
TRANSACTION_COST_BPS = 10
BPS_DENOMINATOR = 10_000
INITIAL_CAPITAL = 100_000
RISK_FREE_RATE = 0.04
TRADING_DAYS_PER_YEAR = 252
BENCHMARK_TICKER = "SPY"

# Plotting
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_DPI = 150
COLOR_STRATEGY = "#2563EB"
COLOR_BENCHMARK = "#6B7280"
COLOR_ACCENT = "#0EA5E9"
COLOR_POSITIVE = "#16A34A"
COLOR_NEGATIVE = "#DC2626"
CONFUSION_CMAP = "Blues"
HIST_BINS = 30

FIGSIZE_EQUITY = (12, 6)
FIGSIZE_DRAWDOWN = (12, 4)
FIGSIZE_IMPORTANCE = (10, 8)
FIGSIZE_MONTHLY_IC = (12, 5)
FIGSIZE_CONFUSION = (5, 4)
FIGSIZE_SIGNAL_DIST = (10, 5)
FIGSIZE_WALK_FORWARD_BASE = (12, 1.5)
WALK_FORWARD_HEIGHT_PER_FOLD = 0.6
WALK_FORWARD_TICKS = 8
SUMMARY_WIDTH = 34

# Reproducibility
RANDOM_SEED = 42
