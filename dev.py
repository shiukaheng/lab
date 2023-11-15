import subprocess
import signal
import sys
from tools import setupwithmeshcat

def signal_handler(sig, frame):
    print('Ctrl-C pressed, shutting down...')
    server_process.terminate()
    sys.exit(0)

# Set up signal handler for Ctrl-C
signal.signal(signal.SIGINT, signal_handler)

# Start meshcat-server in a subprocess
server_process = subprocess.Popen(['meshcat-server'])

# Run your Python code
robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")

# Keep the script running until Ctrl-C is pressed
signal.pause()
