import json
import logging

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "dataload"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "dataload" in active_steps:
            logger.info("Downloading data")
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "dataload"),
                "main",
                env_manager="virtualenv",
                parameters={
                    "url": config["main"]["url"],
                    "sample": config["dataload"]["filename"],
                    "artifact_name": "census.zip",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

if __name__ == "__main__":
    go()