import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import loadmat
from utilities.data_processing import hicToMat, trimMat, contactProbabilities, bootstrap, Sigmoid
from pipeline.models import DenoisingAutoencoder, Classifier

from keras.utils import to_categorical


def trainNN(inputM, targetM, params):
    print('Training autoencoders...')

    odd_dae_model, odd_encoder, _ = DenoisingAutoencoder(inputM, targetM)
    even_dae_model, even_encoder, _ = DenoisingAutoencoder(inputM.T, targetM.T)

    odd_dae_model.fit(inputM[:7000], targetM[:7000], epochs=10, batch_size=32,
                      validation_data=[inputM[7000:], targetM[7000:]])
    even_dae_model.fit(inputM.T[:7000], targetM.T[:7000], epochs=10, batch_size=32,
                       validation_data=[inputM.T[7000:], targetM.T[7000:]])

    odd_encodings = Sigmoid(odd_encoder.predict(inputM))
    even_encodings = Sigmoid(even_encoder.predict(inputM.T))

    odd_dae_model.save(os.path.join(params['dump_dir'], 'odd_chrm_autoencoder.h5'))
    odd_encoder.save(os.path.join(params['dump_dir'], 'odd_chrm_encoder.h5'))

    even_dae_model.save(os.path.join(params['dump_dir'], 'even_chrm_autoencoder.h5'))
    even_encoder.save(os.path.join(params['dump_dir'], 'even_chrm_encoder.h5'))

    return odd_encodings, even_encodings


def train_with_hic(params):
    print('Constructing input matrix')

    inputM = hicToMat(params,
                      prefix='input')

    if params.inter_chromosomal:
        print('Trimming sparse regions...')
        inputM = trimMat(inputM, params.cropIndices)

    print('Computing contact probabilities')
    inputM = contactProbabilities(inputM)

    print('Constructing target matrix')
    if params.input_file != params.target_file:

        targetM = hicToMat(params, prefix='target')

        if params.inter_chromosomal:
            print('Trimming sparse regions...')
            targetM = trimMat(targetM, params['cropIndices'])

        print('Computing contact probabilities')
        targetM = contactProbabilities(targetM)
    else:
        targetM = inputM

    odd_encodings, even_encodings = trainNN(inputM, targetM, params)

    return odd_encodings, even_encodings


"""
This function will bypass the need to reconstruct .mat files of the inter-chromosomal Hi-C matrix
from a raw .hic file.

Use train_with_hic when training SNIPER for the first time. Turn on the -sm flag to save the
matrix as a .mat file, which will save the Hi-C matrix to the output directory specified by the
-dd flag.
"""


def train_with_mat(params):
    print('Using pre-computed .mat files, skipping hic-to-mat')
    inputM = loadmat(params['input_file'])['inter_matrix']
    targetM = loadmat(params['target_file'])['inter_matrix']

    print('Trimming sparse regions from input matrix...')
    inputM = trimMat(inputM, params['cropIndices'])
    print('Computing contact probabilities')
    inputM = contactProbabilities(inputM)

    print('Trimming sparse regions from target matrix...')
    targetM = trimMat(targetM, params['cropIndices'])
    print('Computing contact probabilities')
    targetM = contactProbabilities(targetM)

    trainNN(inputM, targetM, params)
