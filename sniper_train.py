"""
Main training file

Calls helper functions from utilities
"""
import numpy as np
import params
from pipeline.training import train_with_hic

if __name__ == '__main__':
    params = params.Params()

    for chr in range(20, 23, 2):
        params.chr = chr
        encodings = train_with_hic(params)

    print("done")
