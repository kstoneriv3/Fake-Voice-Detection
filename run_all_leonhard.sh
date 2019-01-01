#module load python_gpu/2.7.13 
#module load ffmpeg #pip install --user -r requirements.txt 
run_in_python2_env(){ echo -e 'from __future__ import print_function, unicode_literals, absolute_import, division \nimport os \nimport sys \nfor dir in os.listdir("./src"):\n    sys.path.append(os.path.join("./src",dir))' | cat - $1 > tmp.py python tmp.py $2 $3 rm tmp.py } 

# download data and preprocess it 
#run_in_python2_env ./src/download.py 
#run_in_python2_env ./src/split_normalize_raw_speech.py 

# train VC model and convert voice 
# converted voice is outputted to './data/fake/' 
#run_in_python2_env ./src/conversion/train.py --model_dir='./model/conversion/pretrained' --model_name='pretrained' #run_in_python2_env ./src/conversion/convert.py --model_dir='./model/conversion/pretrained' --model_name='pretrained' 

# train GMM-UBG verification model for two cases: 
# 1. disjoint data for VC and Verification 
# 2. shared data for VC and Verification 
# then plot the score (test statistics, which is log likelihood ratio) 
# model is saved at './model/verification_gmm/' 
#run_in_python2_env ./src/verification_gmm/train_gmm_and_plot_score.py
