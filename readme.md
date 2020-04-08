# Skip-gram with Negative Sampling

This repository contains a python implementation of skip-gram with negative sampling. The code is not optimized as in other libraries (like [gensim](https://radimrehurek.com/gensim/)), but rather focusing on clarity for a pedagogical purpose🐢.

## Setting up
First, install required library. Nothing fancy, just `numpy` for vector calculation and `docopt` for easier command-line handling.  
```bash
pip install -r requirements.txt
```

## Preparing data
 You need tokenized data for the input. The following command downloads the processed Penn Tree Bank (PTB) dataset into `data/ptb`. 
 ```bash
./get_ptb_data.sh
```

## Run training
```bash
python run_train.py data/ptb/ptb.test.txt results/test
```

## See result
```python
from src.skip_gram_with_negative_sampling import SkipGramWithNegativeSampling
model = SkipGramWithNegativeSampling.load('results/test')
model.get_cosine_neighbours('can')
```