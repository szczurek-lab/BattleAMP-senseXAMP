import os
import argparse
import pandas as pd
import sys
from structure_data_generate.cal_pep_des import cal_pep_fromlist


def main(data_dir, out_dir):
    # generate structured data
    files = os.listdir(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    for file in files:
        data_file = os.path.join(data_dir, file)
        data = pd.read_csv(data_file, encoding="utf-8")
        sequence = data['Sequence']
        labels = data['Labels']
        # labels = data['MIC']
        peptides_list = sequence.values.copy().tolist()
        out_path = os.path.join(out_dir, file)
        print("output path: {}".format(out_path))
        cal_pep_fromlist(peptides_list, output_path=out_path, labels=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate structured peptide data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the original datasets')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for the structured datasets')

    args = parser.parse_args()

    main(args.data_dir, args.out_dir)
