import os
import numpy as np
import pandas as pd

from scipy.sparse import vstack


def construct(params, chrom_sizes, hic_dir='.', prefix='hic', hic_res=10000, sizes_file='data/hg19.chrom.sizes'):
    fullSM = None
    chromosome_lengths = chrom_sizes(sizes_file)

    """Span chrms 1, 3, 5, 7... 21"""
    for i in range(1, 3, 2):

        # sparse matrix
        rowSM = None

        """Interactions with even chromosomes"""
        for j in range(2, 4, 2):

            filepath = os.path.join(hic_dir, '{2}_chrm{0}_chrm{1}.txt'.format(i, j, prefix))

            # file = open(filepath,'r')

            txt_data = pd.read_csv(filepath, sep='\t', header=None).values

            nrow = int(chromosome_lengths['chr' + str(i)] / hic_res + 1)
            ncol = int(chromosome_lengths['chr' + str(j)] / hic_res + 1)

            SM = np.zeros((nrow, ncol))

            rows = txt_data[:, 0] / hic_res
            cols = txt_data[:, 1] / hic_res

            if i > j:
                rows = txt_data[:, 1] / hic_res
                cols = txt_data[:, 0] / hic_res

            data = txt_data[:, 2]

            rows = rows.astype(int)
            cols = cols.astype(int)

            try:
                SM[rows, cols] = data
            except IndexError:
                temp = rows.copy()
                rows = cols
                cols = temp
                del temp
                SM[rows, cols] = data

            if rowSM is None:
                rowSM = SM
            else:
                rowSM = np.hstack((rowSM, SM))

        if fullSM is None:
            fullSM = rowSM
        else:
            fullSM = vstack([fullSM, rowSM])

    return fullSM


def construct_chr(params, chrom_sizes, hic_dir='.', prefix='hic', hic_res=10000, sizes_file='data/hg19.chrom.sizes'):
    chromosome_lengths = chrom_sizes(sizes_file)

    filepath = os.path.join(hic_dir, '{1}_chrm{0}_chrm{0}.txt'.format(params.chr, prefix))

    txt_data = pd.read_csv(filepath, sep='\t', header=None).values

    nrow = int(chromosome_lengths['chr' + str(params.chr)] / hic_res + 1)
    ncol = nrow

    SM = np.zeros((nrow, ncol))

    rows = txt_data[:, 0] / hic_res
    cols = txt_data[:, 1] / hic_res

    data = txt_data[:, 2]

    rows = rows.astype(int)
    cols = cols.astype(int)

    SM[rows, cols] = data
    SM[cols, rows] = data

    return SM
