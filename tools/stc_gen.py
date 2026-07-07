from pathlib import Path
from typing import List, Optional, Union

import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class StcProcessor:
    """
    Convert structured datasets to normalized HDF5 files.
    """

    def __init__(
        self,
        label_cols: Optional[List[str]] = None,
    ) -> None:

        self.allseqs = []
        self.norm_parms = None
        self.label_cols = label_cols or []


    def load_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> None:

        self.get_norm_parms(dataset)

        dataset = dataset.copy()

        dataset.drop(
            columns=self.label_cols,
            inplace=True,
        )

        dataset.drop_duplicates(
            ["Sequence"],
            keep="first",
            inplace=True,
        )

        for col in dataset.columns:

            if col == "Sequence":
                continue

            data = dataset[col]

            mean = self.norm_parms[col][0]
            std = self.norm_parms[col][1]

            if std != 0:
                data = (data - mean) / std

            dataset[col] = data

        self.df = dataset


    def get_norm_parms(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        self.norm_parms = {}

        for col in dataframe.columns:

            if (
                col == "Sequence"
                or col in self.label_cols
            ):
                continue

            data = dataframe[col]

            self.norm_parms[col] = [
                data.mean(),
                data.std(),
            ]


    def generate_normed_data(
        self,
        outdir: Union[str, Path],
        fname: str = "normed_structured_data.h5",
    ) -> Path:

        outdir = Path(outdir)

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = outdir / fname

        with h5py.File(output_file, "w") as hf:

            for i in tqdm(range(len(self.df))):

                seq = self.df.iloc[i, 0]

                data = np.array(
                    self.df.iloc[i, 1:]
                ).astype(np.float32)

                hf.create_dataset(
                    seq,
                    data=torch.tensor(data),
                )

        return output_file



def generate_stc(
    dataset_dir: Union[str, Path],
    datafile: str,
    output_dir: Union[str, Path],
    fname: str,
    task: str = "cls",
) -> Path:
    """
    Generate normalized structural HDF5 file.

    Parameters
    ----------
    dataset_dir:
        Directory containing structural CSV.

    datafile:
        CSV filename.

    output_dir:
        Destination HDF5 directory.

    fname:
        Output HDF5 filename.

    task:
        cls or reg.
    """

    dataset_dir = Path(dataset_dir)

    if task == "reg":

        processor = StcProcessor(
            label_cols=[
                "Labels",
                "MIC",
            ]
        )

    elif task == "cls":

        processor = StcProcessor(
            label_cols=[
                "Labels",
            ]
        )

    else:
        raise ValueError(
            "task must be cls or reg"
        )

    dataset = pd.read_csv(
        dataset_dir / datafile
    )

    processor.load_dataset(dataset)

    return processor.generate_normed_data(
        output_dir,
        fname,
    )



def main():

    parser = argparse.ArgumentParser(
        description="Generate stc_info HDF5 file"
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
    )

    parser.add_argument(
        "--datafile",
        required=True,
    )

    parser.add_argument(
        "--fname",
        required=True,
    )

    parser.add_argument(
        "--task",
        choices=["cls", "reg"],
        default="cls",
    )

    parser.add_argument(
        "--output_dir",
        default="./datasets/stc_info",
    )

    args = parser.parse_args()

    generate_stc(
        dataset_dir=args.dataset_dir,
        datafile=args.datafile,
        output_dir=args.output_dir,
        fname=args.fname,
        task=args.task,
    )


if __name__ == "__main__":
    main()