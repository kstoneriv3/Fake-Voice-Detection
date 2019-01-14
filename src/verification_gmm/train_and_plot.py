import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
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
    train_data['Verif_shared'] = train_data['Verif_disjoint']+test_data['train_Conv'][-20:]
    test_data['train_Conv'] = test_data['train_Conv'][-20:]
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
    print('loading data ... ')
    train_data, test_data = load_datasets()
    print('Done.')

    ##############################################
    # train gmm and plot for 2 cases:
    # 1. disjoint data for VC and Verification
    # 2. shared data for VC and Verification 
    ##############################################

    # load pretraind gmm ubg model if it exists
    model_path = './model/verification_gmm/pretrained/gmm_ubg.p'
    if os.path.exists(model_path):
        gmm_ubg = pickle.load(open( model_path, "rb" ))    
    else:
        print('training GMM-UBG model ... '.format(case), end='')
        fit_data = np.concatenate(train_data['ubg'])
        gmm_ubg = GaussianMixture(n_components=2048, covariance_type='diag').fit(fit_data)
        pickle.dump(gmm_ubg, open( model_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
        print('Done.')


    for case in ['disjoint','shared']:

        # load pretrained model if it exists
        model_path = './model/verification_gmm/pretrained/gmm_target_({}_data_for_VC_and_Verif).p'.format(case)
        if os.path.exists(model_path):
            gmm_target = pickle.load(open( model_path, "rb" ))
        else:
            print('training GMM speaker model for {} case ... '.format(case))
            fit_data = np.concatenate([ clip_data for clip_data in train_data['Verif_{}'.format(case)] ])
            gmm_target = GaussianMixture(n_components=256, covariance_type='diag').fit(fit_data)
            pickle.dump(gmm_target, open( model_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            print('Done.')

        # log likelihood ratio    
        def get_LR(samples):
            return + gmm_target.score_samples(samples) - gmm_ubg.score_samples(samples)


        # plot from here
        # for plotting
        names = ['test','ubg_test','fake','train_Conv','validation_Verif']
        if not os.path.exists('./out/plot/'):
            os.makedirs('./out/plot/')

        
        # plot score for small clip (2~10 sec)
        i=0
        one_clip_scores = {name:get_LR(test_data[name][i]) for name in ['test','ubg_test','fake']}
        
        plt.figure(figsize=[10,6])
        for name in one_clip_scores.keys():
            plt.hist(one_clip_scores[name], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake'])
        plt.savefig('./out/plot/score_for_one_small_clip_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
        
        # print mean for each test dataset
        print('average scores of {} data for VC and Verif'.format(case))
        for name in names:
            print(name+':'+' '*(19-len(name))+str(np.mean([get_LR(data).mean() for data in test_data[name]]))) 
        
        # plot score for whole data
        scores = {name: np.concatenate([get_LR(data) for data in test_data[name]])
            for name in names} 
        
        plt.figure(figsize=[10,6])
        for name in names:
            plt.hist(scores[name], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/plot/score_for_whole_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
        
        # plot average score per small clip for whole data
        score_means = {name: np.array([get_LR(data).mean() for data in test_data[name]])
            for name in names}
        
        plt.figure(figsize=[10,6])
        for name in names:
            plt.hist(score_means[name], alpha=0.5, bins=50, density=True,range=[-20,20])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/plot/average_score_per_small_clip_for_whole_({}_data_for_VC_and_Verif).png'.format(case), dpi=1000)
       
        if not os.path.exists('./out/scores/'):
            os.makedirs('./out/scores/')

        # save statistics for explatory analysis
        pickle.dump(one_clip_scores, 
                    open( './out/scores/one_clip_scores_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(scores, 
                    open( './out/scores/scores_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(score_means, 
                    open( './out/scores/score_means_({}_data_for_VC_and_Verif).p'.format(case), "wb" ), 
                    protocol=pickle.HIGHEST_PROTOCOL)

    # just plot the histgram used in report
    for case in ['disjoint']:
        score_means = pickle.load(open( './out/scores/score_means_({}_data_for_VC_and_Verif).p'.format(case), 'rb') )
        plt.figure(figsize=[10,6])
        i = 1
        for name in ['test','fake','ubg_test']:
            plt.hist(score_means[name], alpha=0.5, bins=70, density=True,range=[-20,10])
        plt.legend(['test','fake','universal background'])
        plt.savefig('./out/plot/average_score_per_small_clip_for_test,fake,ubg.svg', format="svg")        
