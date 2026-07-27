#!/usr/bin/env python3
"""
Main entry point for the Hand Gesture Application.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import main

if __name__ == "__main__":
    main()
