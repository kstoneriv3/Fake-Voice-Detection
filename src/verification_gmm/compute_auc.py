import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import pandas as pd


def load_wavs_as_matrices(data_dir):
    filenames = os.listdir(data_dir)
    out = []
    for filename in filenames:
        filepath = os.path.join(data_dir,filename)
        y, sr = librosa.load(filepath)
        mfccs=librosa.feature.mfcc(y,sr)
        out.append(mfccs.T)
    return out


def load_datasets():
    # load test data 
    test_data_dirs = {
        #'train_Conv' :'./data/target/train_conversion/',
        #'test'       :'./data/target/test/',
        #'ubg_test'   :'./data/ubg/test/',
        #'fake'       :'./data/fake'
    }
    voice_dirs = {'fake{}'.format(50*n):'./model/conversion/pretrained/validation_output/converted_B_{}epc'.format(50*n) for n in range(21)}
    test_data_dirs.update(voice_dirs)
    test_data = {}
    for name, data_dir in test_data_dirs.items():
        test_data[name] =load_wavs_as_matrices(data_dir)
        
    return test_data


if __name__ == '__main__':
    
    # get model directory
    parser = argparse.ArgumentParser(description = 'Train GMM models for verification and plot the scores for data.')
    model_dir_default = './model/verification_gmm/pretrained'
    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    argv = parser.parse_args()
    model_dir = argv.model_dir
    
    if not os.path.exists(model_dir):
        raise("no directorty named"+model_dir)

    #load data
    print('loading data ... ')
    test_data = load_datasets()
    print('Done.')

    # load pretraind gmm ubg model if it exists
    model_path = './model/verification_gmm/pretrained/gmm_ubg.p'
    if os.path.exists(model_path):
        gmm_ubg = pickle.load(open( model_path, "rb" ))    
    else:
        raise("no directory:"+model_path)

    for case in ['disjoint']:

        # load pretrained model if it exists
        model_path = './model/verification_gmm/pretrained/gmm_target_({}_data_for_VC_and_Verif).p'.format(case)
        if os.path.exists(model_path):
            gmm_target = pickle.load(open( model_path, "rb" ))
        else:
            raise("no directory:"+model_path)

        # log likelihood ratio    
        def get_LR(samples):
            return + gmm_target.score_samples(samples) - gmm_ubg.score_samples(samples)

        # compute average score per small clip for whole data
        score_means = {name: np.array([get_LR(data).mean() for data in test_data[name]])
            for name in ['fake{}'.format(50*n) for n in range(21)]}
       
        if not os.path.exists('./out/scores/'):
            os.makedirs('./out/scores/')
        # save average scores
        pd.DataFrame(score_means).to_csv('./out/scores/epoch_vs_scores.csv')
    
    # compute auc below
    data_test = pickle.load(open( './out/scores/score_means_(disjoint_data_for_VC_and_Verif).p', "rb" ))['test']
    data= pd.read_csv('./out/scores/epoch_vs_scores.csv', index_col="Unnamed: 0")
    auc = {}
    sigmoid = lambda x:(np.tanh(x)+1)/2.

    for i in range(21):
        data_fake = data['fake{}'.format(50*i)]
        y_true = np.concatenate([np.ones([len(data_test)]), np.zeros([len(data_fake)])])
        y_pred = np.concatenate([
            sigmoid(data_test),
            sigmoid(data_fake)
        ])
        sample_w = np.concatenate([np.ones([len(data_test)])/len(data_test), np.ones([len(data_fake)])/len(data_fake)])
        auc['fake {} epochs'.format(50*i)] = roc_auc_score(y_true,y_pred, sample_weight=sample_w)
    for name,val in auc.items():
        print("{}:\t{}".format(name, val))
