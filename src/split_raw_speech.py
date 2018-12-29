#############################################
# If you are downloading the preprocessed dataset from the google drive, you do not need to use this.
# This code is for splliting original audio files into smaller pieces for training stage.
#############################################
import os
import numpy as np
import librosa

#hand crafted pooling window size
N_POOL = 10400 

def split_file(filepath, output_dir, is_ver):
    
    data, fs = librosa.load(filepath, sr=16000, mono=True, dtype=np.float64)
    
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
    if sound_start[0]>sound_end[0]:
        sound_end_tmp = sound_end[1:]
        sound_end = sound_end_tmp
    
    if len(sound_start)!=len(sound_end):
        sound_start_tmp = sound_start[:-1]
        sound_start = sound_start_tmp

    #return sound_start, sound_end
    
    # save splitted files
    for i in range(len(sound_end)):
        start = sound_start[i]
        end   = sound_end[i]
        tmp   = data[start*N_POOL:end*N_POOL]
        sub_dir = 'train_conversion' if is_ver==True else 'train_verification' if i<len(sound_end)/2 else 'test'
        if not os.path.exists(os.path.join(output_dir,sub_dir)):
            os.makedirs(os.path.join(output_dir,sub_dir))
        out_path = os.path.join(output_dir,sub_dir,filepath.split('/')[-1].replace('.','_{}.'.format(i)))
        librosa.output.write_wav(out_path, tmp, fs)
    

if __name__ == '__main__':
    
    target_dir = './data/target_raw/' 
    output_dir ='./data/target/'
    
    filepathes = [os.path.join(target_dir,filename) for filename in os.listdir(target_dir)] 
    
    split_file(filepathes[0], output_dir='./data/target', is_ver=True)
        
    for filepath in filepathes:
        split_file(filepath, output_dir='./data/target', is_ver=False)
        