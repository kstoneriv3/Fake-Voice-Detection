module load python_gpu/3.6.1 
module load ffmpeg 
pip install --user -r requirements.txt 
pip uninstall tensorflow

# download data and preprocess it 
python ./src/download.py 
python ./src/split_normalize_raw_speech.py 

