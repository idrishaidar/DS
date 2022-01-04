# predict single input
import numpy as np

def predict(model, input):
    predicted = model.predict(input)
    predicted_classes = np.argmax(predicted[0], axis=1)
    predicted_boxes = predicted[1]

    return predicted_classes, predicted_boxes