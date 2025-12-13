# Entry point for Streamlit Cloud deployment
# This file is required by Streamlit Cloud; it imports and runs app.py

import sys
import os

# Ensure app.py can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all content from app.py
# This makes the Streamlit app runnable from streamlit_app.py
exec(open('app.py').read())
