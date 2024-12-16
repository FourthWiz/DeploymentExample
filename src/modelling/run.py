#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import requests
import pickle
from utils.model import ModelLGB, TEXT_COLUMNS
import pandas as pd
import json

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    run = wandb.init(job_type="modelling")
    run.config.update(args)
    wandb.define_metric("ROC", summary="max")

    logger.info(f"Downloading dataset {args.dataset}")

    artifact = run.use_artifact(args.dataset + ":latest")
    artifact_dir = artifact.download()
    data = pd.read_csv(os.path.join(artifact_dir, args.dataset))

    logger.info(f"Initializing model")
    model = ModelLGB(data, TEXT_COLUMNS)

    logger.info(f"Training model and getting metrics")
    model.train_first()
    pr_curve = "pr_curve.png"
    model.calc_metrics(pr_curve)
    run.log({"pr_curve": wandb.Image(pr_curve)})
    run.log({"ROC": model.roc_auc})

    slices = model.sliced_predictions(['education'])
    slices = pd.DataFrame(slices)
    slices.to_csv("slice_output.txt", index=False)
    # with open("slices.txt", "w") as f:
    #     json.dump(slices, f)
    run.log({"slices": wandb.Table(dataframe=slices)})
    
    logger.info("Training the model on full data")
    model.train_total()

    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    run.log_model(path="model.pkl", name=args.artifact_name)
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and save the model")

    parser.add_argument("dataset", type=str, help="Processed data input for the model")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    # parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    # parser.add_argument(
    #     "artifact_description", type=str, help="A brief description of this artifact"
    # )

    args = parser.parse_args()

    go(args)