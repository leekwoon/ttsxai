#! /bin/bash
set -e

echo Downloading the pretrained models ...
gdown https://drive.google.com/uc?id=1BaoynstI8zRKl-39y_WpJ-_czJ8rsRih

unzip pretrained_models.zip
rm pretrained_models.zip

echo 
echo Preparation complete!