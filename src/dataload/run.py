#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import wandb
from utils.download import download

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="download_file")
    run.config.update(args)
    data = download()

    if not os.path.exists("data"):
        os.makedirs("data")
    
    data.to_csv(os.path.join(os.getcwd(), "data", args.sample), index=False)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(os.path.join(os.getcwd(), "data", args.sample))
    artifact.save()
    artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)