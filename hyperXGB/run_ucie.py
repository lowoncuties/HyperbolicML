#!/usr/bin/env python3
"""
Run the UCI dataset experiments with correct Python path setup.
This script ensures the packages can be properly imported.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import and run the actual script
from main_run_UCIE import tune_main

if __name__ == "__main__":
    tune_main()
