#!/bin/bash

# Step 1: prepare datasets for probing
python scripts/make_activation_data.py
python scripts/make_prosody.py
python scripts/make_articulatory_features.py

# for prosody analysis 
python scripts/analyze_prosody.py

# probing examples
python scripts/probe_tacotron2_articulatory_features.py
python scripts/probe_tacotron2_duration.py


