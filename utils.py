import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scheduler import Process
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

def calculate_metrics(processes: List[Process]) -> Dict[str, float]:
    """Calculate various performance metrics for the scheduling results."""
    waiting_times = [p.waiting_time for p in processes]
    turnaround_times = [p.turnaround_time for p in processes]
    burst_times = [p.burst_time for p in processes]
    
    return {
        'avg_waiting_time': np.mean(waiting_times),
        'avg_turnaround_time': np.mean(turnaround_times),
        'max_waiting_time': np.max(waiting_times),
        'max_turnaround_time': np.max(turnaround_times),
        'cpu_utilization': sum(burst_times) / max(p.completion_time for p in processes),
        'throughput': len(processes) / max(p.completion_time for p in processes)
    }

def generate_process_summary(processes: List[Process]) -> pd.DataFrame:
    """Generate a summary DataFrame of process statistics."""
    data = []
    for p in processes:
        data.append({
            'PID': p.pid,
            'Arrival Time': p.arrival_time,
            'Burst Time': p.burst_time,
            'Priority': p.priority,
            'Waiting Time': p.waiting_time,
            'Turnaround Time': p.turnaround_time,
            'Completion Time': p.completion_time,
            'Response Ratio': (p.waiting_time + p.burst_time) / p.burst_time
        })
    return pd.DataFrame(data)

def plot_algorithm_comparison(results: Dict[str, List[Process]]) -> plt.Figure:
    """Create a comparison plot of different scheduling algorithms."""
    metrics = {}
    for algo, processes in results.items():
        metrics[algo] = calculate_metrics(processes)
    
    df = pd.DataFrame(metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=COMPARISON_FIGSIZE)
    fig.suptitle('Algorithm Performance Comparison')
    
    # Plot 1: Average Waiting Time
    sns.barplot(data=df, y='avg_waiting_time', ax=axes[0,0])
    axes[0,0].set_title('Average Waiting Time')
    
    # Plot 2: Average Turnaround Time
    sns.barplot(data=df, y='avg_turnaround_time', ax=axes[0,1])
    axes[0,1].set_title('Average Turnaround Time')
    
    # Plot 3: CPU Utilization
    sns.barplot(data=df, y='cpu_utilization', ax=axes[1,0])
    axes[1,0].set_title('CPU Utilization')
    
    # Plot 4: Throughput
    sns.barplot(data=df, y='throughput', ax=axes[1,1])
    axes[1,1].set_title('Throughput')
    
    plt.tight_layout()
    return fig

def generate_random_processes(num_processes: int) -> List[Process]:
    """Generate a list of random processes for testing."""
    processes = []
    for i in range(num_processes):
        process = Process(
            pid=i+1,
            arrival_time=np.random.uniform(MIN_ARRIVAL_TIME, MAX_ARRIVAL_TIME),
            burst_time=np.random.uniform(MIN_BURST_TIME, MAX_BURST_TIME),
            priority=np.random.randint(MIN_PRIORITY, MAX_PRIORITY + 1)
        )
        processes.append(process)
    return processes

def save_model(model: Any, path: str = MODEL_PATH) -> None:
    """Save the trained model to disk."""
    import joblib
    joblib.dump(model, path)

def load_model(path: str = MODEL_PATH) -> Any:
    """Load the trained model from disk."""
    import joblib
    return joblib.load(path) 