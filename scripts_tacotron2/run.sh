#!/bin/bash

# Step 1: make activation data separately
python scripts/make_activation_data.py

# Step 2: make dataframe for training classifier/generative model
python scripts/make_activation_df.py