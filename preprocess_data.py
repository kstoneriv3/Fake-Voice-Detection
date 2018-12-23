#############################################
# If you are downloading the preprocessed dataset from the google drive, you do not need to use this.
# This code is for splliting original audio files into smaller pieces for training stage.
#############################################

def split_file(filename, output_dir):
    
    data, fs = librosa.load(filename, sr=16000, mono=True, dtype=np.float64)
    
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
        sub_dir = 'train_convert' if i<len(sound_end)/3 else 'train_verification' if i<len(sound_end)/3*2 else 'test'
        if not os.path.exists('{}/{}'.format(output_dir,sub_dir)):
            os.makedirs('{}/{}'.format(output_dir,sub_dir))
        librosa.output.write_wav('{}/{}/{}_{}.wav'.format(output_dir,sub_dir,filename.split('.')[0] ,i), tmp, fs)
    

# for the case of splitting omaba_original_1.wav ~ omaba_original_4.wav

filenames = [
    'filepath/obama_original_1.wav',
    'filepath/obama_original_2.wav',
    'filepath/obama_original_3.wav',
    'filepath/obama_original_4.wav'
]

for filename in filenames:
    split_file(filename, './data/obama')