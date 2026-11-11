from flask import Flask

app = Flask(__name__)

import sys
import os
from pathlib import Path
from io import BytesIO

# Streamlit app code - will be available but may not work in Vercel serverless environment
# The 'app' variable is what Vercel's Flask framework expects

# If running locally with `python app.py`, the Flask app won't be used;
# instead, run with `streamlit run app.py` for full Streamlit functionality

if __name__ == "__main__":
    import streamlit as st
    st.write("Breadfruit Classification System")