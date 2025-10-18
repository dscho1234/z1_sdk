#!/usr/bin/env python3
"""
Setup script for Unitree Z1 SDK
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install


class CMakeExtension(Extension):
    """Extension class for CMake-based builds"""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command for CMake extensions"""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Ensure the extension directory exists
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
            
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # Add platform-specific arguments
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={}'.format(extdir)]
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=Release']
            
        build_args = ['--config', 'Release']
        
        # Add parallel build support
        if hasattr(self, 'parallel') and self.parallel:
            build_args += [f'-j{self.parallel}']
            
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
            
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


class CustomDevelop(develop):
    """Custom develop command that builds the C++ extension"""
    
    def run(self):
        # Build the C++ extension first
        self.run_command('build_ext')
        # Then run the normal develop command
        develop.run(self)


class CustomInstall(install):
    """Custom install command that builds the C++ extension"""
    
    def run(self):
        # Build the C++ extension first
        self.run_command('build_ext')
        # Then run the normal install command
        install.run(self)


def get_long_description():
    """Read the long description from README.md"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Unitree Z1 Robot Arm SDK"


def get_requirements():
    """Get requirements from the project"""
    requirements = [
        "numpy>=1.19.0",
        "gym>=0.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
    ]
    return requirements


# Check if we're building from source or installing pre-built binaries
def should_build_from_source():
    """Determine if we should build from source or use pre-built binaries"""
    # Check if the .so files already exist
    lib_dir = Path(__file__).parent / "lib"
    so_files = list(lib_dir.glob("unitree_arm_interface*.so"))
    return len(so_files) == 0


# Setup configuration
setup_kwargs = {
    "name": "unitree-z1-sdk",
    "version": "1.0.0",
    "author": "Unitree",
    "author_email": "support@unitree.com",
    "description": "Unitree Z1 Robot Arm SDK for Python",
    "long_description": get_long_description(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/unitreerobotics/z1_sdk",
    "packages": find_packages(),
    "package_data": {
        "": ["*.so", "*.pyd", "*.dll"],
        "unitree_arm_sdk": ["*.h", "*.hh"],
    },
    "include_package_data": True,
    "python_requires": ">=3.8",
    "install_requires": get_requirements(),
    "extras_require": {
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "keywords": "robotics, robot-arm, unitree, z1, sdk, control, kinematics",
    "zip_safe": False,
}

# Add CMake extension if building from source
if should_build_from_source():
    setup_kwargs.update({
        "ext_modules": [CMakeExtension('unitree_arm_interface', '.')],
        "cmdclass": {
            'build_ext': CMakeBuild,
            'develop': CustomDevelop,
            'install': CustomInstall,
        },
    })

setup(**setup_kwargs)
