!/bin/bash

model='resnet'

echo "Starting $model model builder"

nohup python model_builder.py Mar12_AVA_MINI256.pt AVA 1 256 resnet &> model_builder.out & 