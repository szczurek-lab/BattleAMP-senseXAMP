import argparse
import pandas as pd


def convert_outputs(input_file: str, output_file: str,
					mode: Literal["cls", "reg"]):
	"""Add sequence id to the inference output file."""
	print("Converting outputs")
	with open(input_file, "r") as file:
		lines = file.readlines()

	sequence_to_id = dict()
	for i in range(0, len(lines), 2):
		idx = lines[i].strip()
		if idx:
			idx = idx.replace(">", "")
			seq = lines[i + 1].strip()
			sequence_to_id[seq] = idx

	output_df = pd.read_csv(output_file, sep="\t")
	output_df["Sequence_id"] = output_df.apply(
		lambda row: sequence_to_id[row["Sequence"]], axis=1
	)
	if mode == "cls":
		output_df = output_df[
			["Sequence_id", "Sequence", "Probability_score", "Prediction"]
		]

	elif mode == "reg":
		output_df["MIC_unit"] = "uM"
		output_df.rename(columns={"MIC": "Log10_MIC"}, inplace=True)
		output_df["MIC"] = output_df.apply(
			lambda row: 10 ** row["Log10_MIC"], axis=1
		)
		output_df = output_df[
			["Sequence_id", "Sequence", "Log10_MIC", "MIC", "MIC_unit"]
		]

	else:
		raise ValueError("Unrecognized inference mode, "
						 "please choose one from (cls, reg)")

	output_df.to_csv(output_file, sep="\t", index=False)
	print("Finished converting outputs")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Convert outputs by adding sequence id column"
	)
	parser.add_argument("-i", "--input_file", dest="input_file", type=str,
						help="Path to the input FASTA file.", required=True)
	parser.add_argument("-o", "--output_file", dest="output_file", type=str,
						help="Path to the output tsv file.", required=True)

	args = parser.parse_args()
	convert_outputs(args.input_file, args.output_file)
