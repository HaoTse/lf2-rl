#!/usr/bin/bash

echo "Checking LF2-Gym files..."
if [ ! -d lf2gym ]; then
    echo "lf2gym/ does not exist, cloning one from Github..."
    git clone https://github.com/HaoTse/lf2gym.git
fi

pushd lf2gym
bash setup.sh
popd
