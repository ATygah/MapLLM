#!/usr/bin/env python3
"""
Script to run the GPT2-small model test.
Usage: python run_gpt2_small_test.py [optimization_level]
  optimization_level: 0 (no optimization) or 1 (skip embeddings and normalizations)
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.tests.gpt2_small_full_test import test_gpt2_small

if __name__ == "__main__":
    # Parse command line arguments
    optimization_level = 0  # Default
    
    if len(sys.argv) > 1:
        try:
            optimization_level = int(sys.argv[1])
            if optimization_level not in [0, 1]:
                print(f"Invalid optimization level: {optimization_level}. Using default (0).")
                optimization_level = 0
        except ValueError:
            print(f"Invalid optimization level: {sys.argv[1]}. Using default (0).")
    
    print(f"Running GPT2-small model test with optimization level {optimization_level}...")
    
    # Run the test with the specified optimization level
    test_gpt2_small(optimization_level=optimization_level)
    
    print("Test completed.") 