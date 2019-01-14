# (train VC model) and convert voice 
# converted voice is outputted to './data/fake/' 
# python ./src/conversion/train.py --model_dir='./model/conversion/new_model' # this takes about a whole day on GTX1080!
python ./src/conversion/convert.py --model_dir='./model/conversion/pretrained' # This uses pretrained model.

# train GMM-UBG verification model for two cases: 
# 1. disjoint data for VC and Verification 
# 2. shared data for VC and Verification 
# then plot the score (test statistics, which is log likelihood ratio) 
python ./src/verification_gmm/train_and_plot.py --model_dir='./model/verification_gmm/pretrained' # This uses pretrained model, training GMM-UBG model takes 4 hours with 40GB memory!

# in case of python2, use 'run_in_python2' instead of 'python'
# run_in_python2(){
#   echo -e 'from __future__ import print_function, unicode_literals, absolute_import, division \nimport os \nimport sys \nfor dir in os.listdir("./src"):\n    sys.path.append(os.path.join("./src",dir))' | cat - $1 > tmp.py 
#   python tmp.py $2 $3 
#   rm tmp.py 
#   } 
# run_in_python2 ./src/conversion/train.py --model_dir='./model/conversion/new_model' --model_name='pretrained'
