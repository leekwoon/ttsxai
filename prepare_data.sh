#! /bin/bash
set -e

mkdir -p $(pwd)/data

echo Downloading the LJ Speech dataset ...
curl -O https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

echo Extracting the files ...
tar -xvf LJSpeech-1.1.tar.bz2

mv LJSpeech-1.1 data/LJSpeech
rm LJSpeech-1.1.tar.bz2

echo 
echo Preparation complete!