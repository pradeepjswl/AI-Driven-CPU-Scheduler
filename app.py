import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scheduler import Process, Scheduler
from data_generator import generate_training_data, save_training_data, load_training_data
from utils import (
    calculate_metrics, generate_process_summary, plot_algorithm_comparison,
    generate_random_processes, save_model, load_model
)
from config import *

# Set page config for better responsiveness
st.set_page_config(
    page_title="AI-Driven CPU Scheduler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for neon theme
st.markdown("""
    <style>
    /* Root variables */
    :root {
        --primary-blue: #0068fd;
        --primary-blue-dim: #3386fd;
        --primary-blue-fog: #cce1ff;
        --primary-purple: #6b57ff;
        --primary-purple-dim: #8979ff;
        --primary-purple-fog: #e1ddff;
        --neon-green: #c8ff00;
        --neon-cyan: #00ffff;
        --neon-magenta: #ff00ff;
        --dark-bg: #0a0a1a;
        --darker-bg: #000000;
        --card-bg: #1a1a2a;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --transition-fast: 300ms;
        --transition-medium: 500ms;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: var(--dark-bg);
        color: var(--text-primary);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 104, 253, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(107, 87, 255, 0.1) 0%, transparent 20%);
    }
    
    /* Neon button styling */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: var(--dark-bg);
        color: var(--neon-cyan);
        border: 2px solid var(--neon-cyan);
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all var(--transition-fast) ease;
        text-shadow: 0 0 5px var(--neon-cyan);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        background-color: var(--neon-cyan);
        color: var(--dark-bg);
        box-shadow: 0 0 15px var(--neon-cyan);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Metric card styling */
    .stMetric {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid var(--neon-green);
        box-shadow: 0 0 15px rgba(200, 255, 0, 0.2);
        transition: all var(--transition-fast) ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 20px rgba(200, 255, 0, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: var(--darker-bg);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--neon-cyan);
        background-color: var(--card-bg);
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all var(--transition-fast) ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--neon-cyan);
        color: var(--dark-bg);
        box-shadow: 0 0 15px var(--neon-cyan);
        border-color: var(--neon-cyan);
    }
    
    /* Header styling */
    h1 {
        color: var(--neon-cyan);
        text-shadow: 0 0 10px var(--neon-cyan);
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        display: inline-block;
    }
    
    h1::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--neon-cyan), transparent);
    }
    
    h2 {
        color: var(--neon-green);
        text-shadow: 0 0 8px var(--neon-green);
        font-size: 1.8rem;
        margin: 1.5rem 0;
    }
    
    h3 {
        color: var(--neon-magenta);
        text-shadow: 0 0 8px var(--neon-magenta);
        font-size: 1.4rem;
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: var(--card-bg);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);
    }
    
    /* Input field styling */
    .stNumberInput input {
        background-color: var(--card-bg);
        color: var(--neon-cyan);
        border: 1px solid var(--neon-cyan);
        border-radius: 5px;
        padding: 0.5rem;
        transition: all var(--transition-fast) ease;
    }
    
    .stNumberInput input:focus {
        box-shadow: 0 0 10px var(--neon-cyan);
        border-color: var(--neon-cyan);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: var(--card-bg);
        border: 1px solid var(--neon-green);
        border-radius: 10px;
        padding: 1rem;
        color: var(--neon-green);
        box-shadow: 0 0 15px rgba(200, 255, 0, 0.2);
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: var(--card-bg);
        border: 1px solid var(--neon-magenta);
        border-radius: 10px;
        padding: 1rem;
        color: var(--neon-magenta);
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
    }
    
    /* Error message styling */
    .stError {
        background-color: var(--card-bg);
        border: 1px solid #ff0000;
        border-radius: 10px;
        padding: 1rem;
        color: #ff0000;
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--darker-bg);
        border-right: 1px solid var(--neon-cyan);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--card-bg);
        color: var(--neon-cyan);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        transition: all var(--transition-fast) ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--neon-cyan);
        color: var(--dark-bg);
        box-shadow: 0 0 15px var(--neon-cyan);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--darker-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neon-cyan);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neon-green);
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .stSpinner {
        animation: pulse 1.5s infinite;
    }
    </style>
""", unsafe_allow_html=True)

def plot_gantt_chart(processes, title="Process Execution Timeline"):
    # Set dark theme for the plot
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=GANTT_CHART_FIGSIZE)
    y_positions = range(len(processes))
    
    # Neon color palette with transparency
    colors = plt.cm.Set3(np.linspace(0, 1, len(processes)))
    colors = np.array([(c[0], c[1], c[2], 0.7) for c in colors])
    
    for i, process in enumerate(processes):
        # Add glow effect
        ax.broken_barh([(process.arrival_time + process.waiting_time, process.burst_time)],
                      (i-0.25, 0.5),
                      facecolors=(colors[i]),
                      edgecolors='#00ffff',
                      linewidth=2,
                      alpha=0.8)
        
        # Add process ID and details with enhanced neon effect
        ax.text(process.arrival_time + process.waiting_time + process.burst_time/2,
                i,
                f'P{process.pid}\n({process.burst_time:.1f})',
                ha='center',
                va='center',
                fontsize=8,
                color='#00ffff',
                bbox=dict(facecolor='#1a1a2a', 
                         edgecolor='#00ffff',
                         alpha=0.8,
                         boxstyle='round,pad=0.5',
                         mutation_scale=0.8))
    
    ax.set_ylim(-1, len(processes))
    ax.set_xlim(0, max(p.completion_time for p in processes))
    ax.set_xlabel('Time Units', color='#00ffff', fontsize=10, labelpad=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, color='#00ffff', linestyle='--')
    
    # Add title with glow effect
    plt.title(title, color='#00ffff', pad=20, fontsize=12, 
              bbox=dict(facecolor='#1a1a2a', 
                       edgecolor='#00ffff',
                       alpha=0.8,
                       boxstyle='round,pad=0.5'))
    
    # Set background color
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')
    
    return fig

def plot_process_distribution(processes):
    # Set dark theme for the plot
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DISTRIBUTION_FIGSIZE)
    
    # Burst time distribution with enhanced styling
    burst_times = [p.burst_time for p in processes]
    sns.histplot(burst_times, bins=10, ax=ax1, color='#00ff00', alpha=0.6)
    ax1.set_title('Burst Time Distribution', color='#00ff00', 
                  fontsize=12, pad=20, bbox=dict(facecolor='#1a1a2a', 
                                                edgecolor='#00ff00',
                                                alpha=0.8,
                                                boxstyle='round,pad=0.5'))
    ax1.set_xlabel('Time Units', color='#00ff00', fontsize=10)
    ax1.set_ylabel('Count', color='#00ff00', fontsize=10)
    ax1.tick_params(colors='#00ff00')
    ax1.grid(True, alpha=0.2, color='#00ff00', linestyle='--')
    
    # Priority distribution with enhanced styling
    priorities = [p.priority for p in processes]
    sns.countplot(x=priorities, ax=ax2, color='#ff00ff', alpha=0.6)
    ax2.set_title('Priority Distribution', color='#ff00ff', 
                  fontsize=12, pad=20, bbox=dict(facecolor='#1a1a2a', 
                                                edgecolor='#ff00ff',
                                                alpha=0.8,
                                                boxstyle='round,pad=0.5'))
    ax2.set_xlabel('Priority Level', color='#ff00ff', fontsize=10)
    ax2.set_ylabel('Count', color='#ff00ff', fontsize=10)
    ax2.tick_params(colors='#ff00ff')
    ax2.grid(True, alpha=0.2, color='#ff00ff', linestyle='--')
    
    # Set background color
    ax1.set_facecolor('#0a0a1a')
    ax2.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')
    
    plt.tight_layout()
    return fig

def run_algorithm_analysis(scheduler, algorithm):
    """Run a specific scheduling algorithm and return results and metrics."""
    if algorithm == "FCFS":
        results = scheduler.fcfs()
    elif algorithm == "SJF":
        results = scheduler.sjf()
    elif algorithm == "Priority":
        results = scheduler.priority()
    else:  # Round Robin
        results = scheduler.round_robin()
    
    metrics = calculate_metrics(results)
    return results, metrics

def main():
    st.title("AI-Driven CPU Scheduler")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Training data generation
        if st.button("Generate Training Data", key="train_data"):
            with st.spinner("Generating training data..."):
                X, y = generate_training_data(TRAINING_SAMPLES)
                save_training_data(X, y)
                st.success("Training data generated successfully!")
        
        # Random process generation
        st.subheader("Generate Random Processes")
        num_processes = st.number_input("Number of Processes", 
                                      min_value=1, 
                                      max_value=MAX_PROCESSES_DISPLAY,
                                      value=5)
        if st.button("Generate Random Processes", key="random_processes"):
            processes = generate_random_processes(num_processes)
            if 'scheduler' not in st.session_state:
                st.session_state.scheduler = Scheduler()
            st.session_state.scheduler.clear_processes()
            for process in processes:
                st.session_state.scheduler.add_process(process)
            st.success(f"Generated {num_processes} random processes!")
        
        # Clear processes
        if st.button("Clear All Processes", key="clear"):
            if 'scheduler' in st.session_state:
                st.session_state.scheduler.clear_processes()
                st.success("All processes cleared!")
    
    # Initialize scheduler
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = Scheduler()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Process Input", "Analysis", "Results", "Algorithm Comparison"])
    
    with tab1:
        st.header("Add Process")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pid = st.number_input("Process ID", min_value=1, value=len(st.session_state.scheduler.processes) + 1)
        with col2:
            arrival_time = st.number_input("Arrival Time", min_value=MIN_ARRIVAL_TIME, value=0.0, step=0.1)
        with col3:
            burst_time = st.number_input("Burst Time", min_value=MIN_BURST_TIME, value=1.0, step=0.1)
        with col4:
            priority = st.number_input("Priority", min_value=MIN_PRIORITY, max_value=MAX_PRIORITY, value=5)
        
        if st.button("Add Process", key="add_process"):
            process = Process(pid=pid, arrival_time=arrival_time, burst_time=burst_time, priority=priority)
            st.session_state.scheduler.add_process(process)
            st.success(f"Process {pid} added successfully!")
    
    with tab2:
        if st.session_state.scheduler.processes:
            st.header("Process Analysis")
            
            # Display current processes
            st.subheader("Current Processes")
            processes_data = generate_process_summary(st.session_state.scheduler.processes)
            st.dataframe(processes_data, use_container_width=True)
            
            # Show process distribution
            st.subheader("Process Distribution Analysis")
            fig_dist = plot_process_distribution(st.session_state.scheduler.processes)
            st.pyplot(fig_dist)
            
            # Show basic statistics
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processes", len(st.session_state.scheduler.processes))
            with col2:
                st.metric("Total Burst Time", f"{sum(p.burst_time for p in st.session_state.scheduler.processes):.2f}")
            with col3:
                st.metric("Average Priority", f"{np.mean([p.priority for p in st.session_state.scheduler.processes]):.2f}")
    
    with tab3:
        if st.session_state.scheduler.processes:
            st.header("Scheduling Results")
            
            if st.button("Run Scheduling", key="run_scheduling"):
                # Train model if not already trained
                if not st.session_state.scheduler.is_trained:
                    try:
                        X, y = load_training_data()
                        st.session_state.scheduler.train_model(X, y)
                        save_model(st.session_state.scheduler.model)
                    except:
                        st.warning("Training data not found. Using default FCFS algorithm.")
                
                # Run selected algorithm
                results = st.session_state.scheduler.run_selected_algorithm()
                selected_algorithm = st.session_state.scheduler.select_best_algorithm()
                
                st.subheader(f"Selected Algorithm: {selected_algorithm.upper()}")
                
                # Display results
                results_data = generate_process_summary(results)
                st.dataframe(results_data, use_container_width=True)
                
                # Display metrics
                metrics = calculate_metrics(results)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Waiting Time", f"{metrics['avg_waiting_time']:.2f}")
                with col2:
                    st.metric("Average Turnaround Time", f"{metrics['avg_turnaround_time']:.2f}")
                with col3:
                    st.metric("CPU Utilization", f"{metrics['cpu_utilization']*100:.1f}%")
                with col4:
                    st.metric("Throughput", f"{metrics['throughput']:.2f}")
                
                # Display Gantt chart
                st.subheader("Process Execution Timeline")
                fig_gantt = plot_gantt_chart(results)
                st.pyplot(fig_gantt)
    
    with tab4:
        if st.session_state.scheduler.processes:
            st.header("Algorithm Comparison")
            
            # Create a copy of the scheduler for analysis
            analysis_scheduler = Scheduler()
            for p in st.session_state.scheduler.processes:
                analysis_scheduler.add_process(p)
            
            # Run all algorithms
            algorithms = ["FCFS", "SJF", "Priority", "Round Robin"]
            all_results = {}
            all_metrics = {}
            
            for algo in algorithms:
                results, metrics = run_algorithm_analysis(analysis_scheduler, algo)
                all_results[algo] = results
                all_metrics[algo] = metrics
            
            # Display comparison metrics
            st.subheader("Performance Metrics Comparison")
            metrics_df = pd.DataFrame(all_metrics).T
            st.dataframe(metrics_df, use_container_width=True)
            
            # Display comparison plots
            st.subheader("Visual Comparison")
            fig_comparison = plot_algorithm_comparison(all_results)
            st.pyplot(fig_comparison)
            
            # Display individual algorithm Gantt charts
            st.subheader("Individual Algorithm Execution Timelines")
            for algo in algorithms:
                st.write(f"### {algo} Timeline")
                fig_gantt = plot_gantt_chart(all_results[algo], f"{algo} Process Execution Timeline")
                st.pyplot(fig_gantt)
            
            # Display detailed analysis
            st.subheader("Detailed Analysis")
            for algo in algorithms:
                with st.expander(f"{algo} Analysis"):
                    results = all_results[algo]
                    metrics = all_metrics[algo]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Waiting Time", f"{metrics['avg_waiting_time']:.2f}")
                    with col2:
                        st.metric("Avg Turnaround Time", f"{metrics['avg_turnaround_time']:.2f}")
                    with col3:
                        st.metric("CPU Utilization", f"{metrics['cpu_utilization']*100:.1f}%")
                    with col4:
                        st.metric("Throughput", f"{metrics['throughput']:.2f}")
                    
                    st.dataframe(generate_process_summary(results), use_container_width=True)

if __name__ == "__main__":
    main() 