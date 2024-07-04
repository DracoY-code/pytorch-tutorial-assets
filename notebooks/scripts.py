"""This script imports the utility functions from the ./scripts directory."""

import os
import sys

# Get the current directory
current_dir = os.getcwd()

# Add the scripts directory to sys.path
scripts_dir = os.path.join(current_dir, '../scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import the utility functions
import plot_utils
import tensor_utils
