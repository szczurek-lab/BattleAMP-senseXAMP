import argparse
import pandas as pd
from typing import Literal


def convert_inputs(fasta_file: str, csv_file: str,
				   mode: Literal["cls", "reg"]) -> None:
	"""Convert inputs from fasta to csv format required by SenseXAMP model."""
	print("Converting inputs")
	with open(fasta_file, "r") as file:
		lines = file.readlines()
	seqs = [x.strip() for x in lines[1::2]]

	target = [None] * len(seqs)
	if mode == "cls":
		data = {"Sequence": seqs, "Labels": target}
	elif mode == "reg":
		data = {"Sequence": seqs, "MIC": target}
	else:
		raise ValueError("Unrecognized inference mode, "
						 "please choose one from (cls, reg)")

	pd.DataFrame(data).to_csv(csv_file, index=False)
	print("Finished converting inputs")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Convert inputs from fasta (required by BattleAMP-benchmark) "
					"to csv (required by SenseXAMP)"
	)
	parser.add_argument("--fasta_file", dest="fasta_file", type=str,
						help="Path to the input FASTA file.", required=True)
	parser.add_argument("--csv_file", dest="csv_file", type=str,
						help="Path to the output csv file.", required=True)

	args = parser.parse_args()
	convert_inputs(args.fasta_file, args.csv_file)
