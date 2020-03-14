#!/bin/bash

model_name=Mar13_Missourian_MINI256
dataset=Missourian
epochs=1
batch_size=256
model=resnet


echo "Starting $model model builder"

nohup python model_builder.py $model_name $datset $epochs $batch_size $model &> model_builder.out & 
