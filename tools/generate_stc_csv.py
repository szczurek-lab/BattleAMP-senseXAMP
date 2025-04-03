import os
import argparse
import pandas as pd
import sys
from structure_data_generate.cal_pep_des import cal_pep_fromlist
from typing import Literal


def main(inpath, outpath, task: Literal["cls", "reg"] = None):
    # generate structured data
    out_dir = os.path.dirname(outpath)
    os.makedirs(out_dir, exist_ok=True)

    data = pd.read_csv(inpath, encoding="utf-8")
    sequence = data['Sequence']
    peptides_list = sequence.values.copy().tolist()
    cal_pep_fromlist(peptides_list, output_path=outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate structured peptide data')
    parser.add_argument('--inputpath', type=str, required=True,)
    parser.add_argument('--outputpath', type=str, required=True)

    args = parser.parse_args()

    main(args.inputpath, args.outputpath)
