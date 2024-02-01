#!/bin/bash

# Step 1: make activation data separately
python scripts/make_activation_data.py

# Step 2: make dataframe for training classifier/generative model
python scripts/make_activation_df.py


python scripts_tacotron2/train_probe_prosody.py --label duration --target_layer conv_0
python scripts_tacotron2/train_probe_prosody.py --label pitch
python scripts_tacotron2/train_probe_prosody.py --label energy