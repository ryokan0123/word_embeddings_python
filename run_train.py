""" train.py.

Usage:
  run_train.py <input_path> <output_path>
  [--force]
  [--model=<model>] [--vec_size=<vec_size>] [--num_iter=<num_iter>] [--alpha=<alpha>]
  [--window_size=<window_size>] [--num_negative_samples=<num_negative_samples>]
  [--min_count=<min_count>]

Options:
  -h --help     Show this screen.
  -f --force     Delete the existing output, if any.
  -i, --num_iter=<num_iter>  The number of iteration. [default: 5].
  -a, --alpha=<alpha>  Initial learning rate. This is linearly decayed during training. [default: 0.025].
  -w, --window_size=<window_size>  Window size. [default: 5].
  -n, --num_negative_samples=<num_negative_samples>  The number of negative samples. [default: 10].
  --model=<model>  The name of model ('sgns', 'sg'). [default: sgns].
  -v, --vec_size=<vec_size>  The size of the vector. [default: 100].
  -m, --min_count=<min_count>  Minimum count for building the vocabulary. [default: 1].
"""
from docopt import docopt
import tqdm
import sys
import shutil
from pathlib import Path

from src.data_processing import DatasetReader, Vocabulary, extract_word_context_pair
from src.data_processing.utils import SubSampling
from src.skip_gram import SkipGram
from src.skip_gram_with_negative_sampling import SkipGramWithNegativeSampling
from src.negative_sampler import FreqBasedNegativeSampler

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_ALPHA = 0.0001

if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)

    output_path = Path(args["<output_path>"])
    if output_path.exists():
        if args['--force']:
            shutil.rmtree(output_path)
        else:
            raise Exception(f"Output path {output_path} already exists!")

    reader = DatasetReader()

    logger.info(f"Reading data from {args['<input_path>']}...")
    tokenized_setences = reader.read(args['<input_path>'])

    logger.info(f"Building vocabulary...")
    vocab = Vocabulary.from_tokens_list(tokenized_setences, min_count=int(args['--min_count']))
    logger.info(f"Vocabulary size: {len(vocab)}")

    if args['--model'] == 'sg':
        model = SkipGram(vocab=vocab, vec_size=int(args['--vec_size']))
        logger.info(f"Train Skip-gram")
    elif args['--model'] == 'sgns':
        vocab_freqs = [vocab.counter[t] for t in vocab.idx2token]
        negative_sampler = FreqBasedNegativeSampler(vocab_freqs=vocab_freqs,
                                                    num_samples=int(args['--num_negative_samples']))
        model = SkipGramWithNegativeSampling(vocab=vocab, vec_size=int(args['--vec_size']),
                                             negative_sampler=negative_sampler)
        logger.info(f"Train Skip-gram with Negative Sampling")

    else:
        raise NotImplementedError

    counter = {vocab.get_idx(t): f for t, f in vocab.counter.items()}
    sub_sampler = SubSampling(counter)

    # convert tokens to ids
    training_indices = [[vocab.get_idx(t) for t in sentence] for sentence in tokenized_setences]

    # these variables are necessary for linear decay for alpha
    num_iter = int(args['--num_iter'])
    total_steps = num_iter * len(training_indices)
    start_alpha = float(args['--alpha'])
    step = 0

    # run training!
    try:
        for iter in range(num_iter):
            logger.info(f"Iteration {iter}")
            for sentence in tqdm.tqdm(training_indices):
                # perform subsampling
                sentence = [t for t in sentence if sub_sampler.is_valid_token(t)]
                alpha = max(start_alpha * (1 - step / total_steps), MIN_ALPHA)
                for w, c in extract_word_context_pair(sentence, int(args['--window_size'])):
                    model.fit(w, c, alpha)
                step += 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    logger.info(f"Saving model to {output_path}...")
    model.save(output_path)
