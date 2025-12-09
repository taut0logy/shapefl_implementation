#!/usr/bin/env python3
"""
Run Edge Aggregator Script
==========================
Entry point for starting a ShapeFL edge aggregator.
"""

import os
import sys

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.edge_aggregator import main

if __name__ == "__main__":
    main()
