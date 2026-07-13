import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Union
from tools.convert_inputs import convert_inputs
from tools.esm_emb_gen import generate_embeddings
from tools.generate_stc_csv import generate_stc_csv
from tools.stc_gen import generate_stc

from model_inference import run_model

import numpy as np
import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# importing model_inference (above) sets SENSEXAMP_ROOT to the source repo root;
# __file__ here points into site-packages (non-editable install), so use the env var.
_CONFIGS_DIR = Path(os.environ["SENSEXAMP_ROOT"]) / "configs"


def _write_fasta(
    sequences: Iterable[str],
    path: Union[str, Path],
) -> None:
    """
    Write sequences to FASTA file.
    """

    path = Path(path)

    with path.open("w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n")
            f.write(f"{seq}\n")


def _filter_sequences(
    sequences: list[str],
):
    """
    Keep only valid SenseXAMP sequences.

    Valid:
    - length 6-25
    - standard amino acids only
    """

    kept = []
    skipped = []

    for seq in sequences:

        seq = seq.upper()

        if len(seq) < 6 or len(seq) > 25:
            skipped.append(seq)
            continue

        if not all(
            aa in STANDARD_AA
            for aa in seq
        ):
            skipped.append(seq)
            continue

        kept.append(seq)

    return kept, skipped



def predict(
    sequences: list[str],
) -> pd.DataFrame:
    """
    Predict AMP activity and MIC values.

    Parameters
    ----------
    sequences:
        List of peptide sequences.

    Returns
    -------
    pandas.DataFrame
    """

    sequences = list(sequences)

    valid_sequences, skipped = _filter_sequences(
        sequences
    )

    if skipped:
        print(
            f"Skipped {len(skipped)} invalid sequences"
        )

    if not valid_sequences:
        return pd.DataFrame(
            columns=[
                "sequence",
                "Prediction",
                "Probability_score",
                "MIC_ecoli",
                "MIC_unit_ecoli",
                "MIC_saureus",
                "MIC_unit_saureus",
            ]
        )


    with TemporaryDirectory(prefix="sensexamp_") as tmp:

        tmp = Path(tmp)

        fasta_file = tmp / "input.fasta"

        _write_fasta(
            valid_sequences,
            fasta_file,
        )

        #
        # 1. Convert FASTA -> CSV
        #

        cls_csv = tmp / "input.cls.csv"
        reg_csv = tmp / "input.reg.csv"

        convert_inputs(
            fasta_file=fasta_file,
            csv_file=cls_csv,
            task="cls",
        )

        convert_inputs(
            fasta_file=fasta_file,
            csv_file=reg_csv,
            task="reg",
        )


        #
        # 2. Generate ESM embeddings
        #

        embedding_dir = tmp / "embeddings"

        embedding_file = generate_embeddings(
            dataset_path=cls_csv,
            output_dir=embedding_dir,
        )


        #
        # 3. Generate structural CSVs
        #

        stc_dataset_dir = tmp / "stc_datasets"

        cls_stc_csv = (
            stc_dataset_dir /
            "input.cls.csv"
        )

        reg_stc_csv = (
            stc_dataset_dir /
            "input.reg.csv"
        )

        generate_stc_csv(
            input_path=cls_csv,
            output_path=cls_stc_csv,
            task="cls",
        )

        generate_stc_csv(
            input_path=reg_csv,
            output_path=reg_stc_csv,
            task="reg",
        )


        #
        # 4. Generate structural HDF5
        #

        stc_info_dir = tmp / "stc_info"

        cls_stc_h5 = generate_stc(
            dataset_dir=stc_dataset_dir,
            datafile="input.cls.csv",
            output_dir=stc_info_dir,
            fname="input.cls.h5",
            task="cls",
        )

        reg_stc_h5 = generate_stc(
            dataset_dir=stc_dataset_dir,
            datafile="input.reg.csv",
            output_dir=stc_info_dir,
            fname="input.reg.h5",
            task="reg",
        )


        print("Preprocessing complete")
        print("CSV exists:", cls_csv.exists())
        print("Embedding exists:", embedding_file.exists())
        print("CLS STC exists:", cls_stc_h5.exists())
        print("REG STC exists:", reg_stc_h5.exists())

        #
        # 5. Run classifier
        #

        cls_output = tmp / "classifier.tsv"

        cls_result = run_model(
            config_path=str(_CONFIGS_DIR / "cls_task" / "battleamp_SenseXAMP.py"),
            task="cls",
            output_path=cls_output,
            datafile=cls_csv,
            embedding_fpath=embedding_file,
            stc_fpath=cls_stc_h5,
        )


        #
        # 6. Run E.coli regression
        #

        ecoli_output = tmp / "ecoli.tsv"

        ecoli_result = run_model(
            config_path=str(_CONFIGS_DIR / "reg_task" / "ecoli_battleamp.py"),
            task="reg",
            output_path=ecoli_output,
            datafile=reg_csv,
            embedding_fpath=embedding_file,
            stc_fpath=reg_stc_h5,
        )


        #
        # 7. Run S.aureus regression
        #

        saureus_output = tmp / "saureus.tsv"

        saureus_result = run_model(
            config_path=str(_CONFIGS_DIR / "reg_task" / "saureus_battleamp.py"),
            task="reg",
            output_path=saureus_output,
            datafile=reg_csv,
            embedding_fpath=embedding_file,
            stc_fpath=reg_stc_h5,
        )

        final = pd.DataFrame({
            "sequence": valid_sequences
        })


        #
        # classifier
        #

        final = final.merge(
            cls_result.rename(
                columns={
                    "Sequence": "sequence"
                }
            )[
                [
                    "sequence",
                    "Prediction",
                    "Probability_score"
                ]
            ].drop_duplicates("sequence"),
            on="sequence",
            how="left",
        )


        #
        # ecoli MIC
        #

        final = final.merge(
            ecoli_result.rename(
                columns={
                    "Sequence": "sequence",
                    "MIC": "MIC_ecoli",
                    "MIC_unit": "MIC_unit_ecoli",
                }
            )[
                [
                    "sequence",
                    "MIC_ecoli",
                    "MIC_unit_ecoli",
                ]
            ].drop_duplicates("sequence"),
            on="sequence",
            how="left",
        )


        #
        # saureus MIC
        #

        final = final.merge(
            saureus_result.rename(
                columns={
                    "Sequence": "sequence",
                    "MIC": "MIC_saureus",
                    "MIC_unit": "MIC_unit_saureus",
                }
            )[
                [
                    "sequence",
                    "MIC_saureus",
                    "MIC_unit_saureus",
                ]
            ].drop_duplicates("sequence"),
            on="sequence",
            how="left",
        )


        return final


def _predict_column(sequences: list[str], column: str) -> np.ndarray:
    """Run predict() and return `column` as a 1-D array aligned to `sequences`.

    predict()'s output has fewer rows and a different order than the input: sequences
    that fail validation (length 6-25, standard amino acids) are dropped, and a valid
    sequence can also be absent if the pipeline produces no row for it. seqme metrics
    expect one score per input sequence in the original order, so we re-align by
    (upper-cased) sequence and fill NaN for any input with no corresponding output row.
    """
    df = predict(sequences)
    lookup = dict(zip(df["sequence"], df[column]))
    return np.array([lookup.get(seq.upper(), np.nan) for seq in sequences], dtype=float)


def predict_amp(sequences: list[str]) -> np.ndarray:
    """AMP classification probability (higher = more likely AMP), one per sequence."""
    return _predict_column(sequences, "Probability_score")


def predict_mic_ecoli(sequences: list[str]) -> np.ndarray:
    """Predicted E. coli MIC in uM (lower = more active), one per sequence."""
    return _predict_column(sequences, "MIC_ecoli")


def predict_mic_saureus(sequences: list[str]) -> np.ndarray:
    """Predicted S. aureus MIC in uM (lower = more active), one per sequence."""
    return _predict_column(sequences, "MIC_saureus")
