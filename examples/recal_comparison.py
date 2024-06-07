#!/bin/python3
#SBATCH --job-name=cal_comp
#SBATCH -t 24:00:00
#SBATCH --mem=30G
#SBATCH --gpus-per-node=v100:1

import argparse
import os
import random
import sys

sys.path.append(os.getcwd())

import numpy as np
import timm
import torch

from datasets import load_dataset
from huggingface_hub import login
from netcal.binning import HistogramBinning, IsotonicRegression
from pathlib import Path

from src.sharpcal.calibration import SharpCal
from src.sharpcal.kernels import Gaussian1D
from src.sharpcal.prediction import get_logits_and_labels_stream
from src.sharpcal.recal import TemperatureScaler
from src.sharpcal.scores import BrierScore, KL

parser = argparse.ArgumentParser(description="Hyperparameters.")
parser.add_argument("--score", dest="score", default="brier", type=str)
parser.add_argument("--bandwidth", dest="bandwidth", default=0.05, type=float)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Calibration setup.
kernel = Gaussian1D(bandwidth=args.bandwidth)
if args.score.lower() == "brier":
    score = BrierScore()
else:
    score = KL()
# Use subsampling for faster computation.
sc = SharpCal(kernel=kernel, score=score, n_points=5000, device=device)

# Load model, dataset and compute logits/labels for dataset.
model_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
dataset = load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)
model = timm.create_model(model_name, pretrained=True).to(device)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
logits, labels = get_logits_and_labels_stream(model, dataset, transforms, cutoff=1000, device=device)  # Cutoff to 1000 for example.

# Calibration set.
cal_cutoff = int(0.2 * len(logits))
cal_logits, cal_labels = logits[:cal_cutoff], labels[:cal_cutoff]
cal_probs = torch.nn.functional.softmax(cal_logits, dim=1)

# Test results.
test_logits, test_labels = logits[cal_cutoff:], labels[cal_cutoff:]
test_probs = torch.nn.functional.softmax(test_logits, dim=1)

# Set up output path.
model_path = f"results/{model_name}/{args.score}_{args.bandwidth}"
Path(model_path).mkdir(parents=True, exist_ok=True)

# Baseline.
sc.plot_cal_curve(test_probs, test_labels.unsqueeze(dim=1), fname=f"{model_path}/baseline.jpg")

# Temperature scaling.
tscale = TemperatureScaler(init_T=1.5, device=device, use_mse=False)
tscale.fit(cal_logits, cal_labels)
print(f"Optimal temperature: {tscale.T}.")
ts_probs = tscale.predict_probs(test_logits)
sc.plot_cal_curve(ts_probs, test_labels.unsqueeze(dim=1), fname=f"{model_path}/temp_scaling.jpg")

# Histogram binning.
binner = HistogramBinning(bins=15)
binner.fit(cal_probs.cpu().numpy(), cal_labels.cpu().numpy())
binner_probs = torch.FloatTensor(binner.transform(test_probs.cpu().numpy())).to(device)
sc.plot_cal_curve(binner_probs, test_labels.unsqueeze(dim=1), fname=f"{model_path}/hist_binning.jpg")

# Isotonic regression.
iso_reg = IsotonicRegression()
iso_reg.fit(cal_probs.cpu().numpy(), cal_labels.cpu().numpy())
iso_probs = torch.FloatTensor(iso_reg.transform(test_probs.cpu().numpy())).to(device)
sc.plot_cal_curve(iso_probs, test_labels.unsqueeze(dim=1), fname=f"{model_path}/iso_regression.jpg")

# MRR.
mean_cal_prob = (cal_probs.argmax(dim=1) == cal_labels).float().mean()
test_preds = test_probs.argmax(dim=1)
mr_probs = (1 - mean_cal_prob) / (test_probs.shape[1] - 1) * torch.ones_like(test_probs)
mr_probs[torch.arange(len(test_probs)), test_preds] = mean_cal_prob
sc.plot_cal_curve(mr_probs, test_labels.unsqueeze(dim=1), fname=f"{model_path}/mrr.jpg")

