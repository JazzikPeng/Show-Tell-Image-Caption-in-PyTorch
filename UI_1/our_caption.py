import torch
import torchvision.transforms as transforms
import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nltk
# from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from model_dropout0 import Encoder, Decoder


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


class CocoDataset(Dataset):
    '''Dataset class for torch.utils.DataLoader'''

    def __init__(self, image_dir, caption_dir, vocab, transform):
        '''
        Parameters:
        ----------
            image_dir: director to coco image
            caption_dir: coco annotation json file path
            vocab: vocabulary wrapper

        '''
        self.image_dir = image_dir
        self.coco = COCO(caption_dir)
        self.keys = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        '''
        Private function return one sample, image, caption
        '''
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        # assert img_file_name.split('.')[-1] == 'jpg'

        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')
        image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # append start and end
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.keys)


def coco_batch(coco_data):
    '''
    create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    '''
    # Sort Descending by caption length
    # print(type(coco_data))
    # print('image:', coco_data[0][0].shape,' | ','captions',coco_data[0][1])
    coco_data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*coco_data)

    # turn images to a 4D tensor with specified batch_size
    images = torch.stack(images, 0)

    # do the same thing for caption. -> 2D tensor with equal length of max lengths in batch, padded with 0
    cap_length = [len(cap) for cap in captions]
    seq_length = max(cap_length)
    # Truncation
    if max(cap_length) > 100:
        seq_length = 100
    targets = torch.LongTensor(np.zeros((len(captions), seq_length)))
    for i, cap in enumerate(captions):
        length = cap_length[i]
        targets[i, :length] = cap[:length]

    return images, targets, cap_length


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def generate_caption(img_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 512
    hidden_size = 512
    num_layers = 3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(img_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)
    # Load vocabulary wraper
    vocab_path = "./data/vocab.pkl"
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build model
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(stateful=False, embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers).to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    encoder.load_state_dict(model["encoder_state_dict"])
    decoder.load_state_dict(model["decoder_state_dict"])

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<<end>>':
            break
    return sampled_caption


if __name__ == '__main__':

    img_path = "./test/000000000139.jpg"
    model_path = "./data/our_model.p"
    print(generate_caption(img_path, model_path))
