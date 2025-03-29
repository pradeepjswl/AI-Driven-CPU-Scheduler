import numpy as np
from scheduler import Process, Scheduler

def generate_random_processes(num_processes: int) -> list[Process]:
    processes = []
    for i in range(num_processes):
        process = Process(
            pid=i+1,
            arrival_time=np.random.uniform(0, 10),
            burst_time=np.random.uniform(1, 10),
            priority=np.random.randint(1, 10)
        )
        processes.append(process)
    return processes

def evaluate_algorithm(scheduler: Scheduler, algorithm: str) -> float:
    if algorithm == "fcfs":
        results = scheduler.fcfs()
    elif algorithm == "sjf":
        results = scheduler.sjf()
    elif algorithm == "priority":
        results = scheduler.priority()
    else:  # round_robin
        results = scheduler.round_robin()
    
    avg_waiting_time = np.mean([p.waiting_time for p in results])
    avg_turnaround_time = np.mean([p.turnaround_time for p in results])
    
    # Combined metric (lower is better)
    return avg_waiting_time + avg_turnaround_time

def generate_training_data(num_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    X = []  # Features
    y = []  # Labels
    
    algorithms = ["fcfs", "sjf", "priority", "round_robin"]
    
    for _ in range(num_samples):
        # Generate random number of processes (5-20)
        num_processes = np.random.randint(5, 21)
        processes = generate_random_processes(num_processes)
        
        # Create scheduler instance
        scheduler = Scheduler()
        for process in processes:
            scheduler.add_process(process)
        
        # Get system metrics
        metrics = scheduler.get_system_metrics()
        X.append(metrics)
        
        # Evaluate all algorithms and select the best one
        best_algorithm = min(algorithms, 
                           key=lambda alg: evaluate_algorithm(scheduler, alg))
        y.append(algorithms.index(best_algorithm))
    
    return np.array(X), np.array(y)

def save_training_data(X: np.ndarray, y: np.ndarray, filename: str = "training_data.npz"):
    np.savez(filename, X=X, y=y)

def load_training_data(filename: str = "training_data.npz") -> tuple[np.ndarray, np.ndarray]:
    data = np.load(filename)
    return data['X'], data['y'] 