"""
Unitree Z1 Robot Arm SDK

This package provides Python bindings for the Unitree Z1 robot arm SDK,
including control interfaces, kinematics, and environment wrappers.
"""

__version__ = "1.0.0"
__author__ = "Unitree"
__email__ = "support@unitree.com"

# Import main components
try:
    from . import unitree_arm_interface
    __all__ = ['unitree_arm_interface']
except ImportError:
    # If the compiled extension is not available, provide a helpful error message
    import warnings
    warnings.warn(
        "unitree_arm_interface module not found. "
        "Please ensure the C++ extension is built and the .so file is available.",
        ImportWarning
    )
    __all__ = []

# Version info
def get_version():
    """Get the version of the SDK"""
    return __version__

def get_author():
    """Get the author information"""
    return __author__

def get_email():
    """Get the email information"""
    return __email__
