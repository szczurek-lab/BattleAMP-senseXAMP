from typing import Dict, List
from xxlimited import Str
import pandas as pd
import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm
import torch


class StcProcessor:
    """
    Convert structured datasets to h5 file
    Args:
        column name of label
    """

    def __init__(self) -> None:
        self.allseqs = []
        self.norm_parms = None

    def load_dataset(self, dataset: pd.DataFrame):
        self.get_norm_parms(dataset)
        dataset.drop_duplicates(['Sequence'], keep='first', inplace=True)
        columns = dataset.columns.tolist()
        # newDataFrame will retain sequence and normed features, drop label cols
        for col in columns:
            if (col == 'Sequence'):
                dataset[col] = dataset[col]
            else:
                data = dataset[col]
                col_mean = self.norm_parms[col][0]
                col_std = self.norm_parms[col][1]
                if col_std != 0:
                    data = ((data - col_mean) / col_std)
                contain_nan = (True in np.isnan(data.values))
                if contain_nan:
                    print("nan col, std is {} mean is: {}".format(data.std(), data.mean()))
                dataset[col] = data
        self.df = dataset
        # print("columns of normed dataframe: {}".format(len(self.df.columns.tolist())))

    def get_norm_parms(self, dataframe):
        self.norm_parms = {}
        columns = dataframe.columns.tolist()
        for col in columns:
            if (col == 'Sequence'):
                continue
            else:
                data = dataframe[col]
                col_mean = data.mean()
                col_std = data.std()
                parms = [col_mean, col_std]
                self.norm_parms[col] = parms

    def generate_normed_data(self, outdir, fname='normed_strutured_data.h5'):
        """
        Generate normed structured data for all sequence and save.
        Args:
            outdir: dir path to save
            fname: filename
        """
        os.makedirs(outdir, exist_ok=True)
        with h5py.File(os.path.join(outdir, fname), 'w') as hf:
            for i in tqdm(range(len(self.df))):
                seq = self.df.iloc[i, 0]
                data = np.array(self.df.iloc[i, 1:]).astype(np.float32)
                data = torch.tensor(data)
                hf.create_dataset(seq, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to calculate stc_info version of dataset')
    parser.add_argument('--dataset_dir', default='./datasets/stc_datasets/AMPlify', help='path to stc_dataset')
    parser.add_argument("--datafile", default="data.csv", help='name of csv with peptides (from stc_datasets)')
    parser.add_argument('--fname', default='AMPlify.h5', help='name of output file name, name it xxx.h5')
    parser.add_argument(
        '--task', type=str, required=False, choices=['cls', 'reg'], default='cls',
        help='Benchmark inference task (cls or reg), required only when using '
             '"inference" mode'
    )
    args = parser.parse_args()
    datadir = args.dataset_dir
    filename = args.datafile
    outdir = './datasets/stc_info'
    outname = args.fname
    if args.task == 'reg':
        processor = StcProcessor()
    else:
        processor = StcProcessor()
    dataset = pd.read_csv(os.path.join(datadir, filename))
    processor.load_dataset(dataset)
    processor.generate_normed_data(outdir, outname)
