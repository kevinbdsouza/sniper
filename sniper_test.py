"""
Main training file

Calls helper functions from utilities
"""
import numpy as np
import params
from pipeline.training import train_with_hic

mode = "test"

if __name__ == '__main__':
    params = params.Params()

    for chr in range(12, 23):
        print("Testing Chromosome {}".format(chr))
        params.chr = chr
        params.mode = mode
        encodings, score = train_with_hic(params)

        np.save(params.dump_dir + '/', 'encoding_' + str(chr) + '.npy', encodings
                )

    print("done")
