import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import loadmat
from utilities.data_processing import DataProcessing
from keras.models import load_model
from pipeline.models import DenoisingAutoencoder


def trainNNchr(inputM, targetM, params, dp_ob):
    print('Training autoencoder')

    train_end = int(0.6 * len(inputM))
    valid_end = train_end + int(0.2 * len(inputM))

    if params.mode == "train":
        dae_model, encoder, _ = DenoisingAutoencoder(inputM, targetM)
        dae_model.fit(inputM[:train_end], targetM[:train_end], epochs=10, batch_size=32,
                      validation_data=[inputM[train_end:valid_end], targetM[train_end:valid_end]])

        encodings = None
        score = None

    elif params.mode == "test":
        dae_model = load_model(os.path.join(params.dump_dir, str(params.chr) + '_autoencoder.h5'))
        encoder = load_model(os.path.join(params.dump_dir, str(params.chr) + '_encoder.h5'))

        score = dae_model.evaluate(inputM[valid_end:], targetM[valid_end:])

        encodings = dp_ob.Sigmoid(encoder.predict(inputM))

    dae_model.save(os.path.join(params.dump_dir, str(params.chr) + '_autoencoder.h5'))
    encoder.save(os.path.join(params.dump_dir, str(params.chr) + '_encoder.h5'))

    return encodings, score


def trainNN(inputM, targetM, params, dp_ob):
    print('Training autoencoders...')

    if params.mode == "train":
        odd_dae_model, odd_encoder, _ = DenoisingAutoencoder(inputM, targetM)
        even_dae_model, even_encoder, _ = DenoisingAutoencoder(inputM.T, targetM.T)

        train_len = int(len(inputM))
        odd_dae_model.fit(inputM[:train_len], targetM[:train_len], epochs=10, batch_size=32,
                          validation_data=[inputM[train_len:], targetM[train_len:]])
        even_dae_model.fit(inputM.T[:train_len], targetM.T[:train_len], epochs=10, batch_size=32,
                           validation_data=[inputM.T[train_len:], targetM.T[train_len:]])
    elif params.mode == "test":
        odd_dae_model = load_model(os.path.join(params.dump_dir, 'odd_chrm_autoencoder.h5'))
        odd_encoder = load_model(os.path.join(params.dump_dir, 'odd_chrm_encoder.h5'))
        even_dae_model = load_model(os.path.join(params.dump_dir, 'even_chrm_autoencoder.h5'))
        even_encoder = load_model(os.path.join(params.dump_dir, 'even_chrm_encoder.h5'))    

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

    os.system("rm {}".format(dp_ob.params.output_txt_path))
    os.system("rm {}".format(dp_ob.params.output_mat_path))

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
