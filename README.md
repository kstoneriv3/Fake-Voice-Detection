# Fake-Voice-Detection

Author: Kei Ishikawa, Xiaoran Chen, Ching Pui WAN, Allan Costa
[Cyclic GAN code](https://github.com/leimao/Voice_Converter_CycleGAN) is by Lei Mao

## Introduction


## Files

```
.
├──src
│   ├── convert.py
│   ├── download.py
│   ├── model.py
│   ├── module.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├──data
│   ├──target_raw (Obama)
│   ├──target (Obama)
│   │   ├─ train_conversion
│   │   ├─ train_verification
│   │   └─ test
│   ├──source
│   │   └─ train_conversion
│   └──ubg
│   │   ├─ train_conversion
│   │   ├─ train_verification
│   │   └─ test
├── figures
├── README.md
```


## Usage

### Before Running Code
```bash
$ cd path/Fake-Voice-Detection/
```

### Download Dataset
Download and unzip datasets and pretrained models.

```bash
$ python ./src/download.py
```

### Split the raw speech
```bash
$ python ./src/split_raw_speech.py
```