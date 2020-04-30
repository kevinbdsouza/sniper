import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import loadmat
from utilities.data_processing import DataProcessing
from pipeline.models import DenoisingAutoencoder
from keras.models import load_model


def trainNNchr(inputM, targetM, params, dp_ob):
    print('Training autoencoder')

    if params.mode == "train":
        dae_model, encoder, _ = DenoisingAutoencoder(inputM, targetM)
        dae_model.fit(inputM[:7000], targetM[:7000], epochs=10, batch_size=32,
                      validation_data=[inputM[7000:], targetM[7000:]])
        score = None

    elif params.mode == "test":
        dae_model = load_model('chrm_autoencoder.h5')
        encoder = load_model('chrm_encoder.h5')

        score = dae_model.evaluate(inputM, targetM)

    encodings = dp_ob.Sigmoid(encoder.predict(inputM))

    dae_model.save(os.path.join(params.dump_dir, 'chrm_autoencoder.h5'))
    encoder.save(os.path.join(params.dump_dir, 'chrm_encoder.h5'))

    return encodings, score


def trainNN(inputM, targetM, params, dp_ob):
    print('Training autoencoders...')

    if params.mode == "train":
        odd_dae_model, odd_encoder, _ = DenoisingAutoencoder(inputM, targetM)
        even_dae_model, even_encoder, _ = DenoisingAutoencoder(inputM.T, targetM.T)

        odd_dae_model.fit(inputM[:7000], targetM[:7000], epochs=10, batch_size=32,
                          validation_data=[inputM[7000:], targetM[7000:]])
        even_dae_model.fit(inputM.T[:7000], targetM.T[:7000], epochs=10, batch_size=32,
                           validation_data=[inputM.T[7000:], targetM.T[7000:]])
    elif params.mode == "test":
        odd_dae_model = load_model('odd_chrm_autoencoder.h5')
        odd_encoder = load_model('odd_chrm_encoder.h5')
        even_dae_model = load_model('even_chrm_autoencoder.h5')
        even_encoder = load_model('even_chrm_encoder.h5')

        odd_score = odd_dae_model.evaluate(inputM, targetM)
        even_score = even_dae_model.evaluate(inputM.T, targetM.T)

    odd_encodings = dp_ob.Sigmoid(odd_encoder.predict(inputM))
    even_encodings = dp_ob.Sigmoid(even_encoder.predict(inputM.T))

    if params.mode == "train":
        odd_dae_model.save(os.path.join(params.dump_dir, 'odd_chrm_autoencoder.h5'))
        even_dae_model.save(os.path.join(params.dump_dir, 'even_chrm_autoencoder.h5'))
        odd_encoder.save(os.path.join(params.dump_dir, 'odd_chrm_encoder.h5'))
        even_encoder.save(os.path.join(params.dump_dir, 'even_chrm_encoder.h5'))

    return odd_encodings, even_encodings


def train_with_hic(params):
    print('Constructing input matrix')

    dp_ob = DataProcessing(params)

    inputM = dp_ob.hicToMat(params,
                            prefix='input')

    if params.inter_chromosomal:
        print('Trimming sparse regions...')
        inputM = dp_ob.trimMat(inputM, params.cropIndices)

    print('Computing contact probabilities')
    inputM = dp_ob.contactProbabilities(inputM)

    print('Constructing target matrix')
    if params.input_file != params.target_file:

        targetM = dp_ob.hicToMat(params, prefix='target')

        if params.inter_chromosomal:
            print('Trimming sparse regions...')
            targetM = dp_ob.trimMat(targetM, params['cropIndices'])

        print('Computing contact probabilities')
        targetM = dp_ob.contactProbabilities(targetM)
    else:
        targetM = inputM

    encodings, score = trainNNchr(inputM, targetM, params, dp_ob)

    return encodings, score


"""
This function will bypass the need to reconstruct .mat files of the inter-chromosomal Hi-C matrix
from a raw .hic file.

Use train_with_hic when training SNIPER for the first time. Turn on the -sm flag to save the
matrix as a .mat file, which will save the Hi-C matrix to the output directory specified by the
-dd flag.
"""


def train_with_mat(params):
    dp_ob = DataProcessing(params)
    print('Using pre-computed .mat files, skipping hic-to-mat')
    inputM = loadmat(params['input_file'])['inter_matrix']
    targetM = loadmat(params['target_file'])['inter_matrix']

    print('Trimming sparse regions from input matrix...')
    inputM = dp_ob.trimMat(inputM, params['cropIndices'])
    print('Computing contact probabilities')
    inputM = dp_ob.contactProbabilities(inputM)

    print('Trimming sparse regions from target matrix...')
    targetM = dp_ob.trimMat(targetM, params['cropIndices'])
    print('Computing contact probabilities')
    targetM = dp_ob.contactProbabilities(targetM)

    trainNN(inputM, targetM, params, dp_ob)
