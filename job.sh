#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=1
#$ -t 1
#$ -e ./logs/
#$ -o ./logs/
. /data/nlp/lunar_pilot_env/bin/activate

mkdir -p ./models/
mkdir -p ./runs/

python main.py