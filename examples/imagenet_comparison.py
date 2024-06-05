#!/bin/python3
#SBATCH --job-name=cal_comp
#SBATCH -t 24:00:00
#SBATCH --mem=30G
#SBATCH --gpus-per-node=A5000:1

import argparse
import os
import pickle
import random
import sys

sys.path.append(os.getcwd())

import numpy as np
import timm
import torch

from datasets import load_dataset
from huggingface_hub import login
from pathlib import Path

from src.sharpcal.calibration import SharpCal
from src.sharpcal.kernels import Gaussian1D
from src.sharpcal.scores import BrierScore, KL

parser = argparse.ArgumentParser(description="Hyperparameters.")
parser.add_argument("--cutoff", dest="cutoff", default=None, type=int)
parser.add_argument("--score", dest="score", default="brier", type=str)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Huggingface read credentials.
login(token=os.environ["HF_READ"])


dataset = load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)
model_names = [
    "resnet50.a1_in1k",
    "efficientnet_b3.ra2_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
]

# Calibration setup.
kernel = Gaussian1D(bandwidth=0.05)
if args.score.lower() == "brier":
    score = BrierScore()
else:
    score = KL()
sc = SharpCal(kernel=kernel, score=score, n_points=5000, device=device)

for model_name in model_names:
    # Model path for storing results.
    model_path = f"results/{model_name}"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Model config.
    model = timm.create_model(model_name, pretrained=True).to(device)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Compute and store results.
    preds, labels = sc.get_preds_and_labels_stream(model, dataset, transforms, cutoff=args.cutoff)
    sc.plot_cal_curve(preds, labels, fname=f"{model_path}/cal_curve_{args.score}.png")

    print(f"Finished {model_name}.")