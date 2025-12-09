#!/usr/bin/env python3
"""
Run Cloud Server Script
=======================
Entry point for starting the ShapeFL cloud server.
"""

import os
import sys

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud.cloud_server import main

if __name__ == "__main__":
    main()
