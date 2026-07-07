import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

import h5py
import pandas as pd
import torch
from tqdm import tqdm

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import esm_project as esm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingProcessor:
    """
    Generate ESM-1b embeddings for protein sequences.
    """

    def __init__(self) -> None:
        self.pretrain_model, _ = esm.pretrained.esm1b_t33_650M_UR50S()

        alphabet = esm.Alphabet.from_architecture("roberta_large")
        self.batch_converter = alphabet.get_batch_converter()

        self.pretrain_model = self.pretrain_model.to(device)
        self.pretrain_model.eval()

        self.all_seqs = []

    def get_seqs_from_datasets(
        self,
        datasets_list: List[Union[str, Path]],
    ):
        """
        Read sequences from CSV datasets.
        """

        sequences = []

        for file in datasets_list:
            df = pd.read_csv(file)
            sequences.extend(df["Sequence"].tolist())

        self.all_seqs = list(set(sequences))

    def generate_embeddings(
        self,
        outdir: Union[str, Path],
        mode: str = "all",
        fname: str = "esm_embeddings.h5",
    ):
        """
        Generate embeddings and save them to HDF5.
        """

        if mode not in ["all", "pooling", "cls_token"]:
            raise ValueError(
                "mode must be one of: all, pooling, cls_token"
            )

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        max_len = 64

        if mode == "all":
            max_len = max(len(seq) for seq in self.all_seqs)
            print(f"Max length: {max_len}")

        output_file = outdir / fname

        with h5py.File(output_file, "w") as hf:

            for seq in tqdm(self.all_seqs):

                data = [(seq, seq)]

                _, _, batch_tokens = self.batch_converter(
                    data,
                    max_length=max_len,
                )

                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():

                    results = self.pretrain_model(
                        batch_tokens,
                        repr_layers=[33],
                        return_contacts=True,
                    )

                token_representations = results["representations"][33]

                if mode == "pooling":
                    embedding = token_representations.mean(1).squeeze(0)

                elif mode == "cls_token":
                    embedding = token_representations[:, 0, :].squeeze(0)

                else:
                    embedding = token_representations.squeeze(0)

                hf.create_dataset(
                    seq,
                    data=embedding.cpu().numpy(),
                )


def generate_embeddings(
    dataset_path: Union[str, Path],
    output_dir: Union[str, Path],
    mode: str = "all",
):
    """
    Public API used by predict.py.

    Parameters
    ----------
    dataset_path:
        CSV containing a Sequence column.

    output_dir:
        Directory where the HDF5 embedding file is written.

    Returns
    -------
    Path
        Generated HDF5 file.
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    embedding_name = (
        dataset_path.stem + ".h5"
    )

    processor = EmbeddingProcessor()

    processor.get_seqs_from_datasets(
        [dataset_path]
    )

    processor.generate_embeddings(
        output_dir,
        mode=mode,
        fname=embedding_name,
    )

    return output_dir / embedding_name


def main():

    parser = argparse.ArgumentParser(
        description="Generate ESM embeddings"
    )

    parser.add_argument(
        "--dataset_path",
        default="./datasets/dummy.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="./datasets/esm_embeddings/all",
    )

    args = parser.parse_args()

    output = generate_embeddings(
        args.dataset_path,
        args.output_dir,
    )

    print(f"Generated embeddings: {output}")


if __name__ == "__main__":
    main()