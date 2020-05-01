"""
Main training file

Calls helper functions from utilities
"""
import params
from pipeline.training import train_with_hic
import numpy as np

mode = "train"

if __name__ == '__main__':
    params = params.Params()

    for chr in range(10, 15):
        print("Training Chromosome {}".format(chr))
        params.chr = chr
        params.mode = mode

        encodings, score = train_with_hic(params)

    print("done")
