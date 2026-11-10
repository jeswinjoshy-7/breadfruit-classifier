#!/bin/bash
cd /home/jeswin/data_augmentation
source .venv/bin/activate
nohup streamlit run streamlit_app/app.py --server.port 8501 --server.headless false --server.address localhost > /tmp/streamlit_app.log 2>&1 &
echo "App started with PID $!"
echo "View at http://localhost:8501"
echo "Check log at /tmp/streamlit_app.log"