#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import requests
from utils.move_extract import extract
from utils.preprocess import preprocess_data

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    run = wandb.init(job_type="preprocess_data")
    run.config.update(args)

    logger.info(f"Downloading artifact {args.filename}")

    artifact = run.use_artifact(args.filename + ":latest")
    artifact_dir = artifact.download()

    #logger.info("Extracting artifact")

    #extract(artifact_dir, args.dataset)

    logger.info(f"Preprocessing data")

    data = preprocess_data(os.path.join(artifact_dir, args.filename))
    data.to_csv('dataset.csv', index=False)

    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(os.path.join(os.getcwd(), "dataset.csv"))
    artifact.save()
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("filename", type=str, help="File location in the archive")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)