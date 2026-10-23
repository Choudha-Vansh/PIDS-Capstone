#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Ubuntu 22.04 setup for the PIDS-Capstone ADAS pedestrian project.
# Sets up a Python venv, system libs, and dependencies, then runs the tests.
#
#   chmod +x setup_ubuntu.sh
#   ./setup_ubuntu.sh                # CPU torch, plain venv  (offline demo / dev)
#   ./setup_ubuntu.sh gpu            # CUDA 12.1 torch (needs an NVIDIA GPU + driver)
#   ./setup_ubuntu.sh cpu ros        # ALSO expose the system ROS (rospy, duckietown_msgs)
#   ./setup_ubuntu.sh gpu ros        #   -> use this on the DUCKIEBOT/ROS laptop
#
# The "ros" flag creates the venv with --system-site-packages so the Duckietown
# ROS Python packages (installed system-wide, NOT on pip) are importable from the
# same env that has torch/ultralytics. Needed for run_car.py's ROS camera/command
# path. Without it the offline demo + stream/dry-run paths still work.
#
# Run this from INSIDE the repo folder (where this script lives).
# ─────────────────────────────────────────────────────────────────────────────
set -e
MODE="${1:-cpu}"
ROS="${2:-}"

echo "==> 1/5  System packages (sudo)"
sudo apt-get update
sudo apt-get install -y \
  git python3 python3-venv python3-pip build-essential \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libxcb-xinerama0 libxkbcommon-x11-0        # the last two: only needed for the PyQt dashboard window

echo "==> 2/5  Python virtual environment"
if [ "$ROS" = "ros" ]; then
  echo "    (--system-site-packages: exposing system ROS / duckietown_msgs)"
  python3 -m venv --system-site-packages .venv
else
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> 3/5  PyTorch ($MODE)"
if [ "$MODE" = "gpu" ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> 4/5  Project dependencies"
pip install -r requirements.txt

echo "==> 5/5  Sanity tests (pure python, no GPU needed)"
python tests/test_decision.py
python tests/test_kalman.py
python tests/test_features.py
python tests/test_cruise_policy.py
python tests/test_safety_stop.py

echo
echo "Done. Activate the env in any new shell with:   source .venv/bin/activate"
echo
echo "Try it:"
echo "  # offline demo on a video (camera path):"
echo "  python run_demo.py --video path/to/demo.mp4 --ego-speed 8.3"
echo "  # headless (no window), writes annotated mp4 + CSV:"
echo "  python run_demo.py --video path/to/demo.mp4 --headless"
echo
echo "Duckiebot live loop (needs the Duckietown ROS env for rospy/duckietown_msgs):"
echo "  python run_car.py --veh <duckiebot-hostname> --show      # dry-run, no motion"
echo "  # camera via stream URL instead of ROS, no ROS needed for a first camera test:"
echo "  python run_car.py --source stream --url http://<bot-ip>:8080/stream --show"
