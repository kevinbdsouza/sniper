"""
Main training file

Calls helper functions from utilities
"""
import numpy as np
import params
from pipeline.training import train_with_hic
import numpy as np

mode = "train"

if __name__ == '__main__':
    params = params.Params()

    for chr in range(16, 23, 2):
        params.chr = chr
        params.mode = mode
        encodings, score = train_with_hic(params)

        np.save(params.dump_dir + '/', 'encoding_' + str(chr) + '.npy', encodings
                )

    print("done")
