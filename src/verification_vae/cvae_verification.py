#from __future__ import absolute_import, division, print_function
from src.verification_vae.cvae_keras import *
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import time
#from keras.models import load_model

#import cvae_train 

def load_wavs_as_matrices(data_dir):
    data_path = os.path.join(data_dir, 'mfccs3.p')
    if os.path.exists(data_path):
        return pickle.load(open( data_path, "rb" ))
          
    filenames = os.listdir(data_dir)
    out = {}
    print(data_dir)
    for filename in filenames:
        if(filename[-1]=='p'):
            continue;
        filepath = os.path.join(data_dir,filename)
        y, sr = librosa.load(filepath)
        #sr=sr/10
        y=y[::3]
        mfccs=librosa.feature.mfcc(y,sr)
        max_pad_len=40
        #print(mfccs.shape[1])
        pad_width = max_pad_len - mfccs.shape[1]
        if(pad_width<=0):
            mfccs=mfccs[:,:max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #print(mfccs.shape)
        out[filename] = mfccs.T
    pickle.dump(list(out.values()), open( data_path, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)  
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
    #train_data['Verif_shared'] = train_data['Verif_disjoint']+test_data['train_Conv']
    return train_data,test_data


if __name__ == '__main__':
    # get model directory
    parser = argparse.ArgumentParser(description = 'Train VAE models for verification and plot the scores for data.')
    model_dir_default = './model/verification_cvae/pretrained'
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
    
    #for case in ['disjoint','shared']:
    for case in ['disjoint']:
        # load pretrained model is it exists
        cvae = {}
        for dataset in ['target','ubg']:
            model_path = os.path.join(model_dir, 'cvaes_{}_{}.p'.format(case,dataset))
            if os.path.exists(model_path) and False:
                #cvae[dataset] = load_model(model_path)
                pass
            else:
                print('training CVAE models for {} case ...'.format(case))
                if(dataset=='target'):
                    cvae['target'] = cvae_training(train_data['Verif_{}'.format(case)])
                else:
                    cvae['ubg']    = cvae_training(train_data['ubg'])
                cvae[dataset].save(model_path)
                
                print('is finished')
            
        def get_LR(samples):
            return + log_prob(cvae['target'],samples) - log_prob(cvae['ubg'],samples)
        
        names = ['test','ubg_test','fake','train_Conv','validation_Verif']
        
        # plot score for small clip (2~10 sec)
        i=0
        plt.hist([get_LR(data) for data in test_data['test']], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.hist([get_LR(data) for data in test_data['ubg_test']], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.hist([get_LR(data) for data in test_data['fake']], alpha=0.5, bins=50, density=True,range=[-50,50])
        plt.legend(['test','universal background', 'fake'])
        
        plt.savefig('./out/score_for_one_small_clip_({}_data_for_VC_and_Verif).png'.format(case))
        
        # print mean for each test dataset
        #print({name: np.mean([get_LR(data).mean() for data in test_data[name]])
        #      for name in names}
        #     ) 
        
        # plot score for whole data
        scores = {name: np.array([get_LR(data) for data in test_data[name]])
            for name in names} 
        for name in names:
            if(name=='test' or name=='fake' or name=='ubg_test'):
                hist,bins=np.histogram(scores[name], bins=20, normed=True)
                bin_centers = (bins[1:]+bins[:-1])*0.5
                plt.plot(bin_centers, hist,'.')
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/score_for_whole_({}_data_for_VC_and_Verif)2.png'.format(case))
        
        # plot average score per small clip for whole data
        score_means = {name: np.array([get_LR(data).mean() for data in test_data[name]])
            for name in names}
        for name in names:
            plt.hist(score_means[name], alpha=0.5, bins=50, density=True,range=[-20,20])
        plt.legend(['test','universal background', 'fake','train_conversion','validation_verification'])
        plt.savefig('./out/average_score_per_small_clip_for_whole_({}_data_for_VC_and_Verif).png'.format(case))
        
        