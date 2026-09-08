#!/bin/bash
# Double-click to launch the live neuro-steered hearing demo in your browser.
# By default it uses recorded EEG + the dataset's two talkers and plays sound.
# Put the KU Leuven data in a folder called "AAD" on your Desktop, or drag a
# different folder onto this file in Terminal to override.
cd "$(dirname "$0")"
DATA_DIR="${1:-$HOME/Desktop/AAD}"
echo "Using data: $DATA_DIR"
echo "Installing requirements (first run only)…"
python3 -m pip install -q -r requirements.txt sounddevice 2>/dev/null
echo "Starting demo — your browser will open. Close this window to stop."
python3 src/live_demo.py --data-dir "$DATA_DIR" --subject S3 --trial 4
