import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
import joblib

@dataclass
class Process:
    pid: int
    arrival_time: float
    burst_time: float
    priority: int
    remaining_time: float = 0
    waiting_time: float = 0
    turnaround_time: float = 0
    completion_time: float = 0

class Scheduler:
    def __init__(self):
        self.processes: List[Process] = []
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
        
    def add_process(self, process: Process):
        self.processes.append(process)
        
    def clear_processes(self):
        self.processes = []
        
    def fcfs(self) -> List[Process]:
        sorted_processes = sorted(self.processes, key=lambda x: x.arrival_time)
        current_time = 0
        
        for process in sorted_processes:
            if current_time < process.arrival_time:
                current_time = process.arrival_time
            process.waiting_time = current_time - process.arrival_time
            process.completion_time = current_time + process.burst_time
            process.turnaround_time = process.completion_time - process.arrival_time
            current_time = process.completion_time
            
        return sorted_processes
    
    def sjf(self) -> List[Process]:
        sorted_processes = sorted(self.processes, key=lambda x: (x.arrival_time, x.burst_time))
        current_time = 0
        completed = []
        ready_queue = []
        
        while sorted_processes or ready_queue:
            while sorted_processes and sorted_processes[0].arrival_time <= current_time:
                ready_queue.append(sorted_processes.pop(0))
                
            if not ready_queue:
                if sorted_processes:
                    current_time = sorted_processes[0].arrival_time
                    ready_queue.append(sorted_processes.pop(0))
                else:
                    break
                    
            ready_queue.sort(key=lambda x: x.burst_time)
            current_process = ready_queue.pop(0)
            
            current_process.waiting_time = current_time - current_process.arrival_time
            current_process.completion_time = current_time + current_process.burst_time
            current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
            current_time = current_process.completion_time
            completed.append(current_process)
            
        return completed
    
    def priority(self) -> List[Process]:
        sorted_processes = sorted(self.processes, key=lambda x: (x.arrival_time, -x.priority))
        current_time = 0
        completed = []
        ready_queue = []
        
        while sorted_processes or ready_queue:
            while sorted_processes and sorted_processes[0].arrival_time <= current_time:
                ready_queue.append(sorted_processes.pop(0))
                
            if not ready_queue:
                if sorted_processes:
                    current_time = sorted_processes[0].arrival_time
                    ready_queue.append(sorted_processes.pop(0))
                else:
                    break
                    
            ready_queue.sort(key=lambda x: -x.priority)
            current_process = ready_queue.pop(0)
            
            current_process.waiting_time = current_time - current_process.arrival_time
            current_process.completion_time = current_time + current_process.burst_time
            current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
            current_time = current_process.completion_time
            completed.append(current_process)
            
        return completed
    
    def round_robin(self, quantum: float = 2.0) -> List[Process]:
        sorted_processes = sorted(self.processes, key=lambda x: x.arrival_time)
        current_time = 0
        completed = []
        ready_queue = []
        
        for process in sorted_processes:
            process.remaining_time = process.burst_time
            
        while sorted_processes or ready_queue:
            while sorted_processes and sorted_processes[0].arrival_time <= current_time:
                ready_queue.append(sorted_processes.pop(0))
                
            if not ready_queue:
                if sorted_processes:
                    current_time = sorted_processes[0].arrival_time
                    ready_queue.append(sorted_processes.pop(0))
                else:
                    break
                    
            current_process = ready_queue.pop(0)
            
            if current_process.remaining_time <= quantum:
                current_process.waiting_time = current_time - current_process.arrival_time
                current_process.completion_time = current_time + current_process.remaining_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_time = current_process.completion_time
                completed.append(current_process)
            else:
                current_process.remaining_time -= quantum
                current_time += quantum
                ready_queue.append(current_process)
                
        return completed
    
    def get_system_metrics(self) -> np.ndarray:
        if not self.processes:
            return np.zeros(5)
            
        metrics = np.array([
            len(self.processes),  # Number of processes
            np.mean([p.burst_time for p in self.processes]),  # Average burst time
            np.std([p.burst_time for p in self.processes]),  # Burst time standard deviation
            np.mean([p.priority for p in self.processes]),  # Average priority
            np.std([p.arrival_time for p in self.processes])  # Arrival time standard deviation
        ])
        return metrics
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True
        
    def select_best_algorithm(self) -> str:
        if not self.is_trained:
            return "fcfs"  # Default to FCFS if model is not trained
            
        metrics = self.get_system_metrics().reshape(1, -1)
        prediction = self.model.predict(metrics)[0]
        
        algorithms = ["fcfs", "sjf", "priority", "round_robin"]
        return algorithms[prediction]
    
    def run_selected_algorithm(self) -> List[Process]:
        algorithm = self.select_best_algorithm()
        
        if algorithm == "fcfs":
            return self.fcfs()
        elif algorithm == "sjf":
            return self.sjf()
        elif algorithm == "priority":
            return self.priority()
        else:  # round_robin
            return self.round_robin() 