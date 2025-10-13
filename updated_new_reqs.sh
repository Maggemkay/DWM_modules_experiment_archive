#!/usr/bin/bash

conda install python==3.12 -y

# Laptop runs torch version 2.5.1
# newest (at time of writing) is 2.9
pip install torch torchvision

conda install tqdm psutil numpy matplotlib tensorboard future filelock setproctitle dataclasses Pillow protobuf -y

pip install wandb gql

conda install ipykernel -y