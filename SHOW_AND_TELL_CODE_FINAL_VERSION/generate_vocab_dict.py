'''
File: generate_vocab_dict.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Saturday, 2018-11-17 14:39
Last Modified: Saturday, 2018-11-17 14:39
--------------------------------------------
Desscription:
'''

# Reference <https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py#L8>


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<<unknown>>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == "__main__":
    import nltk
    from collections import Counter
    from pycocotools.coco import COCO
    import logging
    import numpy as np
    import pickle

    log_level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler("data-preprocess.log")
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    json = "./data/annotations/captions_train2014.json"
    portion = 0.995 
    assert portion != 1
    save_path = "./vocab.pkl"
    # construct coco instance
    # Reference: <https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py#L31>
    coco = COCO(json)
    ids = coco.anns.keys()
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if (i+1) % 5000 == 0:
            print("Tokenization Process: {0:.2f}%.".format((i+1)*100/len(ids)))
            logger.info("Tokenization Process: {0:.2f}%.".format((i+1)*100/len(ids)))
    # Keep the most frequently appeared words
    counts = []
    for _, count in counter.items():
        counts.append(count)
    counts.sort(reverse=True)
    cum_ratio = np.cumsum(counts) / np.sum(counts)
    threshold = counts[np.argmax(cum_ratio > portion)]
    words = []
    for word, count in counter.items():
        if count >= threshold:
            words.append(word)
    words.sort()
    vocab = Vocabulary()
    # for padding purpose
    vocab.add_word('<<padding>>')
    vocab.add_word('<<start>>')
    vocab.add_word('<<end>>')
    vocab.add_word('<<unknown>>')

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)

    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    logger.info("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(save_path))
    logger.info("Saved the vocabulary wrapper to '{}'".format(save_path))
