import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pickle
import argparse
import os

def load_wavs_as_matrices(data_dir):
    filenames = os.listdir(data_dir)
    out = {}
    for filename in filenames:
        filepath = os.path.join(data_dir,filename)
        y, sr = librosa.load(filepath)
        mfccs=librosa.feature.mfcc(y,sr)
        out[filename] = mfccs.T
    return list(out.values())

def load_datasets():
    # load training data and test data 
    train_data_dirs = {
        'Verif_disjoint':'./data/target/train_verification/',
        'ubg'           :'./data/ubg/train_verification/'
    }

    train_data = {}
    for name, data_dir in train_data_dirs.items():
        train_data[name] = load_wavs_as_matrices(data_dir)
    
    test_data_dirs = {
        'train_Conv' :'./data/target/train_conversion/',
        'test'       :'./data/target/test/',
        'ubg_test'   :'./data/ubg/test/',
        'fake'       :'./data/fake'
    }

    test_data = {}
    for name, data_dir in test_data_dirs.items():
        test_data[name] =load_wavs_as_matrices(data_dir)
        
    # make shared data
    test_data['validation_Verif'] = train_data['Verif_disjoint'][-20:]
    train_data['Verif_disjoint']  = train_data['Verif_disjoint'][:-20]
    train_data['Verif_shared'] = train_data['Verif_disjoint']+test_data['train_Conv']
    return train_data,test_data


if __name__ == '__main__':
    
    # get model directory
    parser = argparse.ArgumentParser(description = 'Train GMM models for verification and plot the scores for data.')
    model_dir_default = './model/verification_gmm/pretrained'
    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    argv = parser.parse_args()
    model_dir = argv.model_dir
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #load data
    print('loading data ...')
    train_data, test_data = load_datasets()
    print('is finished.')
    
    ##############################################
    # train gmm and plot for 2 cases:
    # 1. disjoint data for VC and Verification
    # 2. shared data for VC and Verification 
    ##############################################
    
    for case in ['disjoint','shared']:
        # load pretrained model is it exists
        model_path = os.path.join(model_dir, 'gmms_{}.p'.format(case))
        if os.path.exists(model_path):
            gmm = pickle.load(open( model_path, "rb" ))
        else:
            print('training GMM models for {} case ...'.format(case))
            gmm = {}
            gmm['target'] = GaussianMixture(n_components=128,\
                                            covariance_type='diag').fit(np.concatenate(train_data['Verif_{}'.format(case)]))
            gmm['ubg']    = GaussianMixture(n_components=2048,\
                                            covariance_type='diag').fit(np.concatenate(train_data['ubg']))
            pickle.dump(gmm, open( model_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            print('is finished')
            
        def get_LR(samples):
            return + gmm['target'].score_samples(samples) - gmm['ubg'].score_samples(samples)
        
        names = ['test','ubg_test','fake','train_Conv','validation_Verif']
        
        # plot score for small clip (2~10 sec)
        i=0
        plt.hist(get_LR(test_data['test'][i]), alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.hist(get_LR(test_data['ubg_test'][i]), alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.hist(get_LR(test_data['fake'][i]), alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake'])
        plt.savefig('./out/score_for_one_small_clip_({}_data_for_VC_and_Verif).png'.format(case))
        
        # print mean for each test dataset
        print({name: np.mean([get_LR(data).mean() for data in test_data[name]])
              for name in names}
             ) 
        
        # plot score for whole data
        scores = {name: np.concatenate([get_LR(data) for data in test_data[name]])
            for name in names} 
        for name in names:
            plt.hist(scores[name], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/score_for_whole_({}_data_for_VC_and_Verif).png'.format(case))
        
        # plot average score per small clip for whole data
        score_means = {name: np.array([get_LR(data).mean() for data in test_data[name]])
            for name in names}
        for name in names:
            plt.hist(score_means[name], alpha=0.5, bins=50, density=True,range=[-20,20])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/average_score_per_small_clip_for_whole_({}_data_for_VC_and_Verif).png'.format(case))
        
        