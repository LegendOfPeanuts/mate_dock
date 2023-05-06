# Auto Docking

Repo for the auto docking task.

The ```vision``` directory is for computer vision processes.

The ```motion``` directory is for motion control.

```main.py``` is the primary running script.

# Environment

Developed in python 3.9, other versions untested.
- opencv 4.7.0.72
- numpy 1.24.3
- pymavlink 2.4.38

Required libraries:
```pip3 install numpy
pip3 install opencv-python
pip3 install pymavlink
```

# Usage

Run the following command in terminal:
```python3 main.py```

# Procedures for pilot

1. Position the rov directly in front of the dock
2. Start the script
3. Make sure the red button is detected
4. Press ``s`` in the ``Frame`` window and release control from the control station
5. Wait for the ROV to dock
6. Script disengages upon completion (or failure)

Last updated: 06-05-2023