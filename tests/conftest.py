"""
Shared pytest configuration and fixtures.
"""
import os
import sys

# Ensure the project root is on sys.path so all modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
