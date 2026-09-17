# imports
import os
import sys
from pathlib import Path

import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

from predict import predict, OUTPUT_COLUMNS

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = Path(root) / ".." / ".." / "checkpoints"


# my model
def my_model(smiles_list):
    return predict(smiles_list, checkpoints_dir)


# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
assert len(smiles_list) == outputs.shape[0]

# write output in a .csv file
write_out(outputs, OUTPUT_COLUMNS, output_file, np.float32)
