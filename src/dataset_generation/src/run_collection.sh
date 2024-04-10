#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
# Define the number of times to run the Python code
num_runs=50  # Change this to the desired number of runs

# Loop to run the Python code multiple times
for ((i=1; i<=$num_runs; i++)); do
    echo "Running Python code - Iteration $i"
    python /homeL/1wang/workspace/toolee_ws/src/dataset_generation/src/pcd_render_with_hand.py  # Replace with the actual Python script name
    echo "--------------------------"
done
