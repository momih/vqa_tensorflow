import json
import os
import urllib
import zipfile
import re
from collections import Counter
import itertools
import numpy as np
from PIL import Image

data_dir = './data/'


def maybe_download_and_extract():
    """
    Will download and extract the VQA data automatically
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # TODO implement progress bar
    # Downloading the question and answers
    datasets = ["Questions", "Annotations"]
    splits = ["Train", "Val"]
    for data in datasets:
        for split in splits:
            url = "http://visualqa.org/data/mscoco/vqa/{}_{}_mscoco.zip".format(data, split)
            filename = url.split('/')[-1]
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                filepath, _ = urllib.urlretrieve(url, filepath)
                zipfile.ZipFile(filepath, 'r').extractall(data_dir)
                print('Successfully downloaded and extracted ', filename)
    
    
    # Downloading images
    for split in [x.lower() for x in splits]:
        url = "http://msvocds.blob.core.windows.net/coco2014/{}2014.zip".format(split)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.urlretrieve(url, filepath)
            zipfile.ZipFile(filepath, 'r').extractall(data_dir)
            print('Successfully downloaded and extracted ', filename)

class VQADataSet(object):
    """
    Base class for the dataset
    """

    def __init__(self, batch_size, data_dir='./data/', split="train", top_answers=3000,
                 max_ques_len=15):
        self.batch_size= batch_size
        self.data_dir = data_dir
        self.split = split
        self.img_dir = self.data_dir + "{}2014/".format(self.split)
        self.top_answers = top_answers
        self.max_ques_len = max_ques_len
        self._data = self.open_and_preprocess(self.split)
        self.question_to_index = self.map_to_index(top=None, answer=False)
        self.answer_to_index = self.map_to_index(top=self.top_answers)
    
    @property
    def answers(self):
        return (x['answers'] for x in self.data)
    
    @property
    def data(self):
        return self._data
    
    @property
    def questions(self):
        return (x['question'] for x in self._data)
    
    def open_and_preprocess(self, split='train', use_nltk=True):
        questions_filename = self.data_dir + "OpenEnded_mscoco_{0}2014_questions.json"
        answers_filename = self.data_dir + "mscoco_{0}2014_annotations.json"
        
        if use_nltk:
            import nltk
            tokenize = nltk.word_tokenize
        else:
            tokenize = lambda x: x.split(' ')         
        
        questions = self._read_json(questions_filename.format(split))['questions']
        # Answers are present as a list of dicts under the 'annotations' key in the resulting 
        # dictionary when the json file is read
        # The following code reads the json file, then extracts the list of answer dicts
        # And then converts the list into a dict indexed by the question_id
        answers_dict = {x['question_id']:x for x in self._read_json(answers_filename.format(split))['annotations']}
        
        for item in questions:
            question = item['question']
            question = tokenize(question.lower()[:-1])
            
            _id = item['question_id']
            answers = answers_dict.get(_id)['answers']
            # converting answers from list of dicts to just a list of answers without
            # confidence or id
            punc = r'[;>")<!$.%=#*&/+,@\'?(-]\s*'
            answers = [re.sub(punc, ' ', x) for x in [x['answer'] for x in answers]]
            
            item['question'] = question
            item['answers'] = answers      
        
        return questions

    def map_to_index(self, top, answer=True):
        if answer:
            _data = self.answers
        else:
            _data = self.questions

        x = itertools.chain.from_iterable(_data)
        counts = Counter(x)
        sorted_common = (x[0] for x in counts.most_common(top))
        vocab = {word:index for index, word in enumerate(sorted_common, start=1)}
        return vocab            
        
    def encode_into_vector(self):
        for item in self.data:
            q_vec = np.zeros(self.max_ques_len)
            for i, word in item['question'][:self.max_ques_len]:
                mapped_index =  self.question_to_index.get(word, 0)
                q_vec[i] = mapped_index
            
            a_vec = np.zeros(self.top_answers)
            counter = Counter(item['answers'])
            most_freq_ans = counter.most_common(1)
            answer_index = self.answer_to_index.get(most_freq_ans, 0)
            a_vec[answer_index] = 1
            
            item['question'] = q_vec
            item['answers'] = a_vec
    
    def preprocess_image(self, image_id):
        path = '{}COCO_val2014_{:012d}.jpg'.format(self.img_dir, image_id)
        img = Image.open(path)
        img = self._scale_img_to_dim(img, [448])
        img = self._center_crop(img, 299, 299)
        img = self._normalize_img(img.resize((448, 448), Image.ANTIALIAS))
        return img
              
        
    def _normalize_img(self, img):
        img = np.array(img)
        img = img.astype(np.float32) * (1/255.0)
        _mean=[0.485, 0.456, 0.406]
        _std=[0.229, 0.224, 0.225]
        img = (img - _mean)/_std
        return img
        
    def _scale_img_to_dim(self, img, desired_dim):
        w, h = img.size
        if w > h:
            ratio = float(desired_dim)/w
            hsize = int(h*ratio)
            img = img.resize((448, hsize), Image.ANTIALIAS)
        else:
            ratio = float(desired_dim)/h
            wsize = int(w*ratio)
            img = img.resize((wsize, 448), Image.ANTIALIAS)
        return img
        
    def _center_crop(self, im, new_width, new_height):
        width, height = im.size   # Get dimensions
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        
        return im.crop((left, top, right, bottom))           
        
    
    def _read_json(self, file):
        with open(file, 'r') as f:
            x = json.load(f)
        return x
    

vqa = VQADataSet(12, './data/', 'val')
a = vqa.create_vocab()