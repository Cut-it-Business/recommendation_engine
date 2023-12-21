import pickle
import clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
import numpy as np
import io

target_path ='../model/sorted_classes.pkl'
clsf_path = '../model/clip_emb_logreg.pickle'

class Recommended:
    def __init__(self, clsf_path,target_path):
        self.classifier = pickle.load(open(clsf_path, 'rb'))
        self.target = pickle.load(open(target_path, 'rb'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device('cpu'):
            print('Warning! GPU not available')
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)

    def recommended(self, image):
        '''
        image: jpeg file?
        '''
        # image = Image.open(io.BytesIO(image)) # image = Image.open(image) #???
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image_input)
        outputs = self.classifier.predict_proba(image_features.detach().numpy())
        probs_ind = outputs[0].argsort().tolist()[::-1]
        state = self.target
        probs = [round(x, 5) for x in outputs[0][probs_ind].tolist()]
        response = dict(zip([state[i] for i in probs_ind], probs))
        return response

    def batch_recommended(self, images):
        '''
        images: List of jpeg files?
        '''
        # images = [Image.open(io.BytesIO(img)) for img in images] #???

        image_feature_list = [self.clip_model.encode_image(self.preprocess(img).unsqueeze(0).to(self.device)) for img in images]
        outputs_list = [self.classifier.predict_proba(img_ftr.detach().numpy()) for img_ftr in image_feature_list]
        outputs = np.array(outputs_list).mean(axis=0)
        probs_ind = outputs[0].argsort().tolist()[::-1]
        state = self.target
        probs = [round(x, 5) for x in outputs[0][probs_ind].tolist()]
        response = dict(zip([state[i] for i in probs_ind], probs))
        return response
