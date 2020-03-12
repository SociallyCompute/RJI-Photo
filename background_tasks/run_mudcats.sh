!/bin/bash

model='resnet'

echo "Starting $model model builder"

nohup python model_builder.py AVA 1 256 resnet &> model_builder.out & 