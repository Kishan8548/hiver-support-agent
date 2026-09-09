"""
conftest.py
-----------
Pytest configuration for the hiver-support-agent test suite.
Ensures project root is in sys.path so `src` imports work without installation.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
