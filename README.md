

# Setup

## Option 1: Install as Python Package (Recommended)

```bash
# Install the package in development mode
cd ~/Workspace/z1_sdk
pip install -e .

# Build the C++ extension (if needed)
python setup.py build_ext

# Verify installation
python -c "import unitree_arm_sdk; print('Package installed successfully!')"
```

## Option 2: Manual Setup

```bash
cd ~/Workspace/z1_contoller
mkdir -p build && cd build
cmake ..
make -j 

cd ~/Workspace/z1_sdk
mkdir -p build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python)
make -j

# check build
ls ../lib/unitree_arm_interface*.so

# set python path
export PYTHONPATH=$PYTHONPATH:$(realpath ../lib)
or 
cp ../lib/unitree_arm_interface*.so ../examples_py/
```


```
# Example

```bash

cd ~/Workspace/z1_contoller/build
./z1_ctrl


cd ~/Workspace/z1_sdk
# Action chunking
python examples_py/example_gym_env_cartesian_cmd_rtc.py 

# image, robot joint data collection
python examples_py/example_free_drive.py

```

