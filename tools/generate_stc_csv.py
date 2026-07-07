from pathlib import Path
from typing import Literal, Union
import argparse

import pandas as pd

from structure_data_generate.cal_pep_des import cal_pep_fromlist


def generate_stc_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    task: Literal["cls", "reg"] = "cls",
) -> None:
    """
    Generate structural feature CSV required by SenseXAMP.

    Parameters
    ----------
    input_path:
        CSV file containing the Sequence column.

    output_path:
        Destination structural CSV.

    task:
        Inference task:
        - cls: classification
        - reg: regression
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        input_path,
        encoding="utf-8",
    )

    sequences = data["Sequence"].tolist()
    labels = data["Labels"]

    if task == "reg":

        mic = data["MIC"]

        cal_pep_fromlist(
            sequences,
            output_path=str(output_path),
            labels=labels,
            mic_results=mic,
        )

    elif task == "cls":

        cal_pep_fromlist(
            sequences,
            output_path=str(output_path),
            labels=labels,
        )

    else:
        raise ValueError(
            "task must be either 'cls' or 'reg'"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Generate structured peptide data"
    )

    parser.add_argument(
        "--inputpath",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--outputpath",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--task",
        choices=["cls", "reg"],
        default="cls",
    )

    args = parser.parse_args()

    generate_stc_csv(
        input_path=args.inputpath,
        output_path=args.outputpath,
        task=args.task,
    )


if __name__ == "__main__":
    main()