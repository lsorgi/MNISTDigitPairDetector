# MNISTDigitPairDetector

## Installation 

1. Download [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).
  
2. Create a new environment:
    ```
    conda create -n [ENV_NAME] python=3.7
    conda activate [ENV_NAME]
    ```
   
5. Install MNISTPairDetector
    ```
    cd ~
    git clone https://github.com/lsorgi/MNISTDigitPairDetector.git
    cd MNISTPairDetector
    pip install -e .
    ```

## Train model 
   
```
cd ~/MNISTPairDetector
python ./bin/run_training.py -c ./cfg/cfg.json -o ./models 
```

Logs are available on Tensorboard

```
tensorboard --logdir=./models
```

