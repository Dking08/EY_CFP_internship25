#!/usr/bin/env python3
"""
Test just the multi-algorithm models to check if feature extraction works
"""

import pandas as pd
import numpy as np
from testcase import AccuracyTester

def test_algorithms_only():
    print("=== TESTING ALGORITHM MODELS ONLY ===")
    
    # Initialize tester
    tester = AccuracyTester()
    tester.load_data_and_models()
    
    # Test just the algorithm comparison
    tester.test_all_algorithms_holdout('2024-01-01')

if __name__ == "__main__":
    test_algorithms_only()
