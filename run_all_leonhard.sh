run_in_python2(){
  echo -e 'from __future__ import print_function, unicode_literals, absolute_import, division \nimport os \nimport sys \nfor dir in os.listdir("./src"):\n    sys.path.append(os.path.join("./src",dir))' | cat - $1 > tmp.py 
  python tmp.py $2 $3 
  rm tmp.py 
  } 

# train VC model and convert voice 
# converted voice is outputted to './data/fake/' 
# in case of python2, use 'run_in_python2' instead of 'python'
# run_in_python2 ./src/conversion/train.py --model_dir='./model/conversion/pretrained' --model_name='pretrained'
python ./src/conversion/train.py --model_dir='./model/conversion/pretrained' --model_name='pretrained'
python ./src/conversion/convert.py --model_dir='./model/conversion/pretrained' --model_name='pretrained' 

# train GMM-UBG verification model for two cases: 
# 1. disjoint data for VC and Verification 
# 2. shared data for VC and Verification 
# then plot the score (test statistics, which is log likelihood ratio) 
# model is saved at './model/verification_gmm/' 
python ./src/verification_gmm/train_gmm_and_plot_score.py
