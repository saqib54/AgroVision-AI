# Entry point for Streamlit Cloud deployment
# This file imports and runs the main app from app/app.py
import runpy
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Run the main app
runpy.run_path(os.path.join(os.path.dirname(__file__), 'app', 'app.py'), run_name='__main__')
