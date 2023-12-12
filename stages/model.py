import pickle
import clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

target_path ='sorted_classes.pkl'
clsf_path = 'clip_emb_logreg.pickle'

class Recommended:
    def __init__(self, clsf_path):
        self.classifier = pickle.load(open(clsf_path, 'rb'))
        self.target = pickle.load(open(target_path, 'rb'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device('cpu'):
            print('Warning! GPU not available')
        self.model, self.preprocess = clip.load('ViT-B/32', device)

    def recommended(self, image):
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image_input)
        outputs = classifier.predict_proba(image_features.detach().numpy())
        probs_ind = outputs[0].argsort().tolist()[::-1]
        state = self.target
        probs = [round(x, 5) for x in outputs[0][probs_ind].tolist()]
        response = dict(zip([state[i] for i in probs_ind], probs))
        return response
