# AI
## Installation

You must have Python 3 installed before installation.

To install the module, use:
```bash
git clone https://github.com/Dima-Ryabinin/AI
cd AI
pip install -r requirements.txt
```

For ***Unix*** systems, you can also do:
```bash
chmod +x ./main.py
```
To run script via `./main.py` instead of `python main.py`

## Usage
### Working with a python script occurs through the passed arguments.

Argument `-t` or `--train` starts neural network training (default dataset is _train_).
```bash
python main.py --train
```

Argument `-p` or `--predict` starts the prediction of object classes from the dataset.
```bash
python main.py --predict
```

Additional arguments:
* Use `-e <count>` or `--epochs-count <count>` to set the number of epochs to train (Default: 50).
* Use `-b <batch size>` or `--batch-size <batch size>` to set the batch size (Default: 32).
* Use `-m <model name>` or `--model <model name>` to set the model name (Default: model).
* Use `-d <dataset name>` or `--dataset <dataset name>` to set the dataset name (Default for train: train, Default for predict: test)
