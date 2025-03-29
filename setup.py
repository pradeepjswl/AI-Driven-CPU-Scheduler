import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("Successfully installed all requirements!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "streamlit",
        "matplotlib",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", ", ".join(missing_packages))
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["data", "models"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    print("Setting up AI-Driven CPU Scheduler...")
    
    # Check Python version
    check_python_version()
    
    # Create necessary directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check dependencies
    if not check_dependencies():
        print("Error: Some dependencies are missing. Please run 'pip install -r requirements.txt' manually.")
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    print("1. Open a terminal")
    print("2. Navigate to the project directory")
    print("3. Run: streamlit run app.py")
    print("\nThe application will open in your default web browser.")

if __name__ == "__main__":
    main() 