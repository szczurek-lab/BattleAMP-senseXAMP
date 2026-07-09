from pathlib import Path
import os
import time

import pandas as pd
import torch

from contextlib import contextmanager
import json

from utils import Config, Logger
from Ampmm_base.runner import Runner

# The SenseXAMP configs are copied into a temp dir before import (mmcv-style
# Config), so they cannot locate the repo via __file__. Expose the repo root
# through an env var they can read instead.
os.environ.setdefault("SENSEXAMP_ROOT", str(Path(__file__).resolve().parent))

@contextmanager
def temporary_custom_json(
    config_path,
    datafile,
    embeddings_fpath,
    stc_fpath,
):
    """
    Create temporary custom.json required by SenseXAMP configs.
    """

    config_dir = Path(config_path).parent

    custom_file = config_dir / "custom.json"

    backup = None

    if custom_file.exists():
        backup = custom_file.read_text()

    try:
        with open(custom_file, "w") as f:

            json.dump(
                {
                    "data_fpath": str(datafile),
                    "embedding_fpath": str(embeddings_fpath),
                    "stc_fpath": str(stc_fpath),
                },
                f,
                indent=2,
            )

        print(custom_file.read_text())

        yield

    finally:
        if backup is not None:
            custom_file.write_text(backup)

        else:
            custom_file.unlink(missing_ok=True)


# TODO: Delete this function later, use run_model instead
def run_inference(
    config_path: str,
    task: str,
    output_path: str,
    datafile: str,
    embeddings_fpath: str,
    stc_fpath: str,
) -> pd.DataFrame:
    """
    Run SenseXAMP inference without torchrun.
    """

    if task not in ["cls", "reg"]:
        raise ValueError(
            "task must be cls or reg"
        )


    #
    # Build configuration
    #

    with temporary_custom_json(
        config_path,
        datafile,
        embeddings_fpath,
        stc_fpath,
    ):
        cfg = Config.fromfile(config_path)

        cfg.data["test"]["datafile"] = str(datafile)
        cfg.data["test"]["embeddings_fpath"] = str(embeddings_fpath)
        cfg.data["test"]["stc_fpath"] = str(stc_fpath)


        #
        # Create work directory
        #

        cfg.work_dir = str(
            Path(cfg.work_dir)
            /
            time.strftime(
                "%Y-%m-%d_%H:%M:%S"
            )
        )

        Path(cfg.work_dir).mkdir(
            parents=True,
            exist_ok=True,
        )


        logger = Logger(
            cfg.work_dir
        )


        #
        # Use GPU 0 or CPU
        #

        device = (
            0
            if torch.cuda.is_available()
            else -1
        )

        #
        # Create runner
        #

        print("DATA EXISTS:", Path(datafile).exists())
        print("EMB EXISTS:", Path(embeddings_fpath).exists())
        print("STC EXISTS:", Path(stc_fpath).exists())

        print("CREATING RUNNER")

        runner = Runner(
            cfg,
            logger,
            device,
            "inference",
        )

        print("RUNNER CREATED")


        #
        # Run inference
        #

        runner.inference(
            output_path,
            task,
        )

        print("INFERENCE FINISHED")

    return pd.read_csv(
        output_path,
        sep="\t",
    )


def run_model(
    config_path: str,
    task: str,
    output_path: str,
    datafile: str,
    embedding_fpath: str,
    stc_fpath: str,
):
    """
    Generic SenseXAMP inference runner.
    """

    config_path = Path(config_path)

    # Configs expect custom.json next to them
    with temporary_custom_json(
        config_path,
        datafile,
        embedding_fpath,
        stc_fpath,
    ):

        cfg = Config.fromfile(config_path)


        # Extra safety, overwrite paths
        cfg.data["test"]["datafile"] = str(datafile)
        cfg.data["test"]["embeddings_fpath"] = str(embedding_fpath)
        cfg.data["test"]["stc_fpath"] = str(stc_fpath)


        cfg.work_dir = str(
            Path(cfg.work_dir)
            /
            "inference"
        )

        Path(cfg.work_dir).mkdir(
            parents=True,
            exist_ok=True,
        )


        logger = Logger(
            cfg.work_dir
        )


        runner = Runner(
            cfg,
            logger,
            0,
            "inference",
        )


        runner.inference(
            str(output_path),
            task,
        )


        df = pd.read_csv(
            output_path,
            sep="\t",
        )


        if task == "reg":
            df["MIC_unit"] = "uM"

            df = df[
                [
                    "Sequence",
                    "MIC",
                    "MIC_unit",
                ]
            ]

        else:
            df = df[
                [
                    "Sequence",
                    "Probability_score",
                    "Prediction",
                ]
            ]


        return df