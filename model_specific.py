""" module for model specific concretisation of abstract interfaces """
import os

import pandas as pd
from tools.converter import Converter, InputConverter
from tools.inference import Inferencer


class ConcreteConverter(Converter):
    def process_file(self, filepath: str, output_filename: str):
        pass

class ConcreteInferencer(Inferencer):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model """
        command = f"sh inference.sh {filepath} {output_filename}"
        print(command)
        os.system(command)


class ConcreteInputConverter(InputConverter):
    def process_file(self, filepath: str, output_filename: str):
        """ implement for specific model
        base format is fasta"""
        with open(filepath) as fp:
            lines = fp.readlines()

        seqs = [x.strip() for x in lines[1::2]]
        labels = [None] * len(seqs)
        pd.DataFrame({"Sequence": seqs, "Labels": labels}).to_csv(output_filename)


