#!/bin/bash

if [ -d "./data/" ]
then
    echo "data/ directory already exists"
else
    mkdir data/
    echo "data/ directory created"
fi

if [ -f "./data/fma_metadata.zip" ]
then
    echo "fma_metadata.zip exists."
    if [ -d "./data/fma_metadata" ]
    then
        echo "fma_metadata.zip already extracted."
    else
        7za x fma_metadata.zip
        echo "fma_metadata.zip extracted."
    fi
else
    echo "Downloading fma_metadata.zip..."
    curl -o ./data/fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
    7za x fma_metadata.zip
    echo "fma_metadata.zip extracted."
fi

if [ -f "./data/fma_small.zip" ]
then
    echo "fma_small.zip exists."
    if [ -d "./data/fma_small" ]
    then
        echo "fma_small.zip already extracted."
    else
        7za x fma_small.zip
        echo "fma_small.zip extracted."
    fi
else
    echo "Downloading fma_small.zip..."
    curl -o ./data/fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
    echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_small.zip" | sha1sum -c -
    7za x fma_small.zip
    echo "fma_small.zip extracted."
fi