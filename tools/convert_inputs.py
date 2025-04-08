import argparse
import pandas as pd
import os
from typing import Literal


def convert_inputs(fasta_file: str, csv_file: str) -> None:
	"""Convert inputs from fasta to csv format required by SenseXAMP taskl."""

	if not os.path.exists(fasta_file):
		print(f"Error: The file '{fasta_file}' does not exist.")
		return

	print("Converting inputs")
	with open(fasta_file, "r") as file:
		lines = file.readlines()
	seqs = [x.strip() for x in lines[1::2]]
	data = {'Sequence': seqs}

	df = pd.DataFrame(data)
	df = df.loc[df['Sequence'].str.len() <= 25]
	df.to_csv(csv_file, index=False)
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
