# This file calls beam_search() and generate image caption
# This version is used to generate caption using our own model and train again on repo's file

import torch
import torchvision.transforms as transforms
import os
import sys
import pickle
import numpy as np
#wimport nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from generate_vocab_dict import Vocabulary
from data_loader import CocoDataset, coco_batch
from pycocotools.coco import COCO
import argparse
from model_V2_dropout0 import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
       #  print('iamge shape1', image.size)
        image = transform(image).unsqueeze(0)
       #  print('image shape2', image.shape)
    return image


def main(args):
	# Image preprocessing
	# In generation phase, we need should random crop, just resize
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

    # Load vocabulary wraper
    with open(args.vocab_path, 'rb') as f:
    	vocab = pickle.load(f)

    # Build model
    encoder = Encoder(embed_size=args.embed_size).eval()
    decoder = Decoder(stateful=False, embed_size=args.embed_size, hidden_size=args.hidden_size, vocab_size=len(vocab), num_layers=args.num_layers).to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)

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
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    print(sampled_caption)
    image = Image.open(args.image)
#    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    PATH = '/home/ubuntu/final_project/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default=PATH + 'model/encoder-1-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=PATH + 'model/decoder-1-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default=PATH + 'data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=3, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)





