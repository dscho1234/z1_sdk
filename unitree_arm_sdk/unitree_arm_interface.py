"""
Python wrapper for the Unitree Z1 robot arm interface.

This module provides a Python interface to the compiled C++ extension.
"""

import sys
import os
from pathlib import Path

# Add the lib directory to the Python path to find the compiled extension
lib_dir = Path(__file__).parent.parent / "lib"
if lib_dir.exists():
    sys.path.insert(0, str(lib_dir))

try:
    # Try to import the compiled extension
    import unitree_arm_interface
    from unitree_arm_interface import *
except ImportError as e:
    # If import fails, provide a helpful error message
    raise ImportError(
        f"Failed to import unitree_arm_interface: {e}\n"
        f"Please ensure:\n"
        f"1. The C++ extension is built (run 'python setup.py build_ext')\n"
        f"2. The .so file exists in the lib/ directory\n"
        f"3. The lib/ directory is in your Python path"
    ) from e

__all__ = [
    'ArmFSMState',
    'LowlevelState', 
    'CtrlComponents',
    'Z1Model',
    'ArmInterface',
    'postureToHomo',
    'homoToPosture',
]
