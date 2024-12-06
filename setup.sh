#!/bin/bash

# Step 1: Initialize Conda (only needs to be done once)
conda init

# Step 2: Create a new conda environment
conda create -n voice-chat-env python=3.9 -y

# Step 3: Activate the new environment
source activate voice-chat-env

# Step 4: Install the dependencies from requirements.txt
pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
