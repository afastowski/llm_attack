#!/bin/bash

for dataset in triviaqa squad nq; do
        for version in 0 1 2 3; do
            echo "Starting with dataset: $dataset"
            python3 uncertainty.py -m mistral -d $dataset -v $version

        echo "Finished with dataset: $dataset"
    done
done
echo "DONE!"
