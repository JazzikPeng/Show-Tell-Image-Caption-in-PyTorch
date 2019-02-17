# Code for Show and Tell: Show and Tell: A Neural Image Caption Generator
This doc explains the functionality of each Python file.

# data_loader.py
```python
class CocoDataset(Dataset):
    def __init__(self, image_dir, caption_dir, vocab,transform):
    def __getitem__(self, idx):
    def __len__(self):
    def showImg(sef, idx):

def coco_batch(coco_data):
```
- This `CocoDataset()` can be passed into `torch.utils.data.DataLoader()` to create a compatiable Pytorch dataloader.
- The `coco_batch()` function can be passed into `collate_fn` in `torch.utils.data.DataLoader()`. `coco_batch()` returns images, targets, and caption length in type of `torch.LongTensor`.

# resize.py
```python
resize_image(image, size)
resize_images(image_dir, output_dir, size)
main()
```
This file is resize all training images in `./data/train2014` to (256, 256) and save it to `./data/resized2014/`. We resize all images, so we can have consistent inputs.  

- `resize_image()` takes a `PIL.Image` objetct and `size` object as inputs. This returns a resized image, with `Image.ANTIALIAS` flag.
- `resize_images()` resize all images in `image_dir` and save it to `output_dir`.
- `main()` calls `resize_images()` function.

# generate\_vocab_dict.py
```python
class Vocabulary(object):
    '''Simple vocabulary wrapper.'''
```

This file generates vocabulary and add <\<padding>>, <\<start>>, <\<end>>, <\<unknown>>, token from captions in training dataset.
This file saves `vocab.pkl` file in the current directory.

- `Vocabulary()` class contains two dictionaries: `word_to_index` and `index_to_word`. Those two dictionaries help us to map captions to index, vice versa.

# model\_V2_dropout0.py
```python
class Encoder(nn.Module):
    def __init__(self, attention=False, embed_size=256, encoded_image_size=14)
    def forward(self, images)

class StatefulLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size)
    def reset_state(self)
    def forward(self, x)

class LockedDropout(nn.Module):
    def __init__(self, dropout=0)
    def reset_state(self)
    def forward(self, x, train=True)

class LSTMBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout=0)
    def reset_state(self)
    def forward(self, x, train=True)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20, stateful=True, dropout=0)
    def reset_state(self)
    def forward(self, features, captions, lengths, train=True)
    def sample(self, features, states=None)

```
This file defines our model. 

- `Encoder()` uses the pre-trained `resnet152`. We fixed all previously trained parameters and only fine-tune on the last fully connect layer and batch normalization layer. The default dropout rate is set to 0.
- `StatefulLSTM()` is the same as the one we used on HW7, 8. It uses a LSTMCell and locked dropout method.
- `LockedDropout()` the dropout mask is fixed throughout the batch. It's call in `LSTMBlock()`.
- `Decoder()` have two form; one using stateful LSTM with locked dropout and the other uses pytorch `nn.LSTM` function. Our best model is trained using pytorch `nn.LSTM` function.

# train.py
`Train.py` imports `Encoder`, `Decoder` from `model_V2_dropout0.py`, `CocoDataset` and `coco_batch` from `data_loader.py`, and `Vocabulary` from `generate_vocab_dict.py`.

This is the training file. Our training process saves a model for every epoch. Our code can load the pre-trained model state and continue training or tuning.
The model used cross entropy loss to train and Adam optimizer for backward propagations All Hyper-parameters is defined at the beginning of the `main()` function.

# generate\_caption_V2.py
This file load trained encoder and decoder and generates output vectors, then the output is mapped to `vocab.idx2word[]` dictionary to generate caption.
This python takes an image path as argument and prints caption.

```bash
python generate_caption_V2.py --image='png/example.png'
```

# bleu.py
```python
def get_image_name(jsonPath):
def bleu_score(input_imgs_path, generated_captions, name_caption_frame):
```
- `get_image_name()` function takes a path to `captions_val2014.json` and returns a data frame with image names and corresponded captions. Each validation image has 5 human captions.
- `bleu_score()` compute bleu score of generated caption using 5 human captions as references and theoretical bleu score by taking the average of bleu score using each human caption as generated caption compares with other 4 human captions as references.

# model_bleu.py
```python
def load_image(image_path, transform=None):
def generate_caption(image_path, vocab, encoder, decoder, transform):
def test(args):
```
This file imports get_image_name and bleu_score from bleu.py.

- `load_image()` function takes a image path and load specified image and convert it to RGB format.
- `generate_caption()` uses the same logic as `generate_caption_V2.py`, but it only returns caption as list of tokens.
- `test()` iterate through all images in `./val2014` and computes the average generated bleu score and average theoratical score. We used `pandas.apply()` to parallelized this process to improve time complexity.

```bash
# testing BLEU-4
python model_bleu.py --eval='eval' 
# training BLEU-4
python model_bleu.py --eval='train'
```