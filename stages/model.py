from ultralytics import YOLO


class Recommended:

    def __init__(self, model_path):
        self.model = YOLO(model_path, task='classify')

    def recommended(self, image):
        res = self.model.predict(image)
        probs_ind = res[0].probs.data.argsort().tolist()[::-1]
        state = res[0].names
        probs = [round(x, 5) for x in res[0].probs.data[probs_ind].tolist()]
        response = dict(zip([state[i] for i in probs_ind], probs))
        return response
