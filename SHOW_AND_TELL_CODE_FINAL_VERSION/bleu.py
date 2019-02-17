import json
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# get imagename dataframe reference, jsonPath is the validation set json file path
def get_image_name(jsonPath):
    # python 3
    with open(jsonPath, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    dataImage = pd.DataFrame.from_dict(data['images'])
    dataAnnotations = pd.DataFrame.from_dict(data['annotations'])
    dataName = pd.merge(dataImage,dataAnnotations, left_on='id',right_on='image_id')
    dataName = dataName[['file_name','caption']]
    return dataName

# name_caption_frame = get_image_name(jsonPath)

# define bleu score calculation, imgpath should be like /example/example.jpg, captions should be like ['<start>','example','example','<end>']
def bleu_score(input_imgs_path, generated_captions, name_caption_frame):
    imgName = input_imgs_path.split('/')[-1]
    captions = list(name_caption_frame[name_caption_frame['file_name']==imgName]['caption'])
#    print(captions)
    references = []
    for i in range(5):
        temp = nltk.word_tokenize(captions[i].lower())
        references.append(temp)
    candidates = generated_captions[1:-1]   
    generated_score = sentence_bleu(references, candidates, smoothing_function=SmoothingFunction().method4)
    theoratical_score = 0
    for i in range(5):
        theoratical_score += sentence_bleu(references[:i]+references[i+1:], references[i], smoothing_function=SmoothingFunction().method4)
        #print(references[:i]+references[i+1:],theoratical_score)
    theoratical_score /= 5.0
    return generated_score,theoratical_score
