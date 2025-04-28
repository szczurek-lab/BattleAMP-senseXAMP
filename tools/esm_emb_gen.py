import argparse
import numpy as np
import pandas as pd
import torch
import h5py
import os
import esm_project as esm
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingProcessor:
    """
    Generate esm embeddings for all proteins.
    """
    def __init__(self) -> None:
        self.pretrain_model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
        alphabet = esm.Alphabet.from_architecture("roberta_large")
        self.batch_converter = alphabet.get_batch_converter()
        self.pretrain_model = self.pretrain_model.to(device)
        self.all_seqs = []

    def get_seqs_from_datasets(self, datasets_list: List[str]):
        """
        Get all sequences from list of datasets
        Args:
            datasets_list: List[String]
        """
        for file in datasets_list:
            df = pd.read_csv(file)
            # print(df)
            self.all_seqs.extend(df['Sequence'].values)
        self.all_seqs = set(self.all_seqs)

    def get_seqs_from_list(self, seq_list: List[str]):
        self.all_seqs = seq_list

    def generate_embeddings(self, outdir, mode='all', fname='esm_embeddings.h5'):
        """
        Generate embeddings for all sequence.
        Args:
            outdir: path to embedding file save
            mode: 
                all or pooling.
            fname: name of embedding file
        """
        assert (mode == 'all') or (mode == 'pooling') or (mode == 'cls_token')
        os.makedirs(outdir, exist_ok=True)
        self.max_len = max(len(seq) for seq in self.all_seqs)
        max_len = 64
        if mode == 'all':
            max_len = self.max_len
            # print("Max length: {}".format(self.max_len))

        with h5py.File(os.path.join(outdir,fname), 'w') as hf:
            for seq in self.all_seqs:
                data = [(seq,seq)]
                _, _, batch_tokens = self.batch_converter(data, max_length=max_len) 
                batch_tokens = batch_tokens.to(device)
                try:
                    with torch.no_grad():
                        results = self.pretrain_model(batch_tokens, repr_layers=[33], return_contacts=True)
                except:
                    print(seq)
                token_representations = results["representations"][33]
                if mode == 'pooling':
                    embedding = token_representations.mean(1).squeeze(0)  # [1280]
                elif mode == 'cls_token':
                    embedding = token_representations[:,0,:].squeeze(0)  # cls token
                else: # mode = 'all' (SenseXAMP use this type)
                    embedding = token_representations.squeeze(0)          # [676,1280]
                    embedding = embedding.cpu().numpy()
                    embedding = embedding.astype('half')  # Reduce precision to save space
                    chunk = 23 if len(seq) > 23 else 2
                    hf.create_dataset(seq, data=embedding, compression="gzip", compression_opts=9, shuffle=True,
                                      chunks=(chunk, 1280))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to calculate esm_embeddings version of datasets')
    parser.add_argument('--dataset_path', default='./datasets/dummy.csv', help='path to raw csv dataset')
    args = parser.parse_args()
    processor = EmbeddingProcessor()
    dataset_path = args.dataset_path
    dataset_name = os.path.split(dataset_path)[-1]
    embedding_name = f"{os.path.splitext(dataset_name)[0]}.h5"
    datasets_list = [dataset_path]
    processor.get_seqs_from_datasets(datasets_list)
    processor.generate_embeddings('./datasets/esm_embeddings/all', mode='all', fname=embedding_name)