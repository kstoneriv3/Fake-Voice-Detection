#############################################
# If you are downloading the preprocessed dataset from the google drive, you do not need to use this.
# This code is for splliting original audio files into smaller pieces for training stage.
#############################################
import os
import numpy as np
import librosa

#hand crafted pooling window size
N_POOL = 10400 

def split_file(filepath, output_dir):
    #laod, normalize volume
    data, fs = librosa.load(filepath, sr=16000, mono=True, dtype=np.float64)
    data = data/(data.var()/0.005)**0.5
    
    # get the absolute size of the signal
    # then smooth (ave pool) the signal 
    data_abs = np.abs(data)
    avepool_data_abs = data_abs[:data_abs.shape[0]//N_POOL*N_POOL].reshape([-1,N_POOL]).mean(axis=1)
        
    # get the start and the end of the utterances
    cutoff_points = np.int16(avepool_data_abs<0.005)    
    sound_start = np.zeros_like(cutoff_points)
    sound_end   = np.zeros_like(cutoff_points)
    for i in range(len(cutoff_points)-1):
        sound_start[i] = cutoff_points[i] - cutoff_points[i+1]>0
        sound_end[i+1] = cutoff_points[i] - cutoff_points[i+1]<0

    sound_start = np.nonzero(sound_start)[0] 
    sound_end   = np.nonzero(sound_end)[0]
    
    # make sure that head and the tail of the file is quiet
    # so that file can be well splitted
    if sound_start[0]>=sound_end[0]:
        sound_end = sound_end[1:]
    
    if sound_start[-1]>=sound_end[-1]:
        sound_start = sound_start[:-1]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # save splitted files
    for i in range(len(sound_end)):
        start = sound_start[i]
        end   = sound_end[i]
        tmp   = data[start*N_POOL:end*N_POOL]
        out_path = os.path.join(output_dir,filepath.split('/')[-1].replace('.','_{}.'.format(i)))
        librosa.output.write_wav(out_path, tmp, fs)

if __name__ == '__main__':
    
    target_dir = './data/target_raw/' 
    
    filepathes = [os.path.join(target_dir,filename) for filename in os.listdir(target_dir)] 
    filepathes.sort()
    
    print('splitting raw speech data ...')
    split_file(filepathes[0], output_dir='./data/target/train_conversion')
        
    for filepath in filepathes[4:7]:
        split_file(filepath, output_dir='./data/target/train_verification')
        
    for filepath in filepathes[1:4]:
        split_file(filepath, output_dir='./data/target/test')
    print('is finished.')
