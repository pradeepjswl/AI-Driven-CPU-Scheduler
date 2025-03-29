# Application Configuration

# Scheduling Algorithm Parameters
ROUND_ROBIN_QUANTUM = 2.0
MAX_PRIORITY = 10
MIN_PRIORITY = 1

# Process Generation Parameters
MIN_BURST_TIME = 0.1
MAX_BURST_TIME = 10.0
MIN_ARRIVAL_TIME = 0.0
MAX_ARRIVAL_TIME = 10.0

# Training Data Parameters
TRAINING_SAMPLES = 1000
MIN_PROCESSES = 5
MAX_PROCESSES = 20

# Visualization Parameters
GANTT_CHART_FIGSIZE = (12, 6)
DISTRIBUTION_FIGSIZE = (15, 5)
COMPARISON_FIGSIZE = (15, 5)

# UI Parameters
MAX_PROCESSES_DISPLAY = 100
REFRESH_RATE = 1  # seconds

# File Paths
TRAINING_DATA_PATH = "data/training_data.npz"
MODEL_PATH = "models/scheduler_model.joblib" 