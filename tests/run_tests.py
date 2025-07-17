#!/usr/bin/env python3
"""
Test runner for the huggingface project.
Run this script to execute all unit tests using pytest.
"""

import sys
import os
import subprocess

if __name__ == '__main__':
    # Run pytest with the current directory as the test directory
    test_dir = os.path.dirname(__file__)
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        test_dir, 
        '-v',  # verbose output
        '--tb=short'  # shorter traceback format
    ], capture_output=False)
    
    # Exit with the same code as pytest
    sys.exit(result.returncode)
