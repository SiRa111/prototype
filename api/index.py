import os
import sys

# Ensure the root directory is in the path to import server.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app
