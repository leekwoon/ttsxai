#! /bin/bash
set -e

mkdir -p $(pwd)/pretrained_models

echo Downloading the pretrained models ...
gdown https://drive.google.com/uc?id=1BaoynstI8zRKl-39y_WpJ-_czJ8rsRih

rm pretrained_models.zip

echo 
echo Preparation complete!