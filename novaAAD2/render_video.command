#!/bin/bash
# Double-click to render the shareable demo video (results/demo_video.mp4).
cd "$(dirname "$0")"
DATA_DIR="${1:-$HOME/Desktop/AAD}"
python3 -m pip install -q -r requirements.txt 2>/dev/null
python3 render_demo.py --data-dir "$DATA_DIR" --subject S3 --trial 4
echo "Done -> results/demo_video.mp4"
