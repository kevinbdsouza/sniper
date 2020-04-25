"""
Main training file

Calls helper functions from utilities
"""
import numpy as np
import sys
from scipy.io import loadmat
import params
from utilities.input import get_params
from pipeline.training import train_with_hic, train_with_mat

if __name__ == '__main__':
    params = params.Params()

    for chr in range(20, 23, 2):
        params.chr = chr
        odd_encodings, even_encodings = train_with_hic(params)

    print("done")
