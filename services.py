from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import json
class ModelService:
    
    def __init__(self,tp):
        if tp=="mobilenetV2":
            self.model = load_model('./mobilenet_classify.h5', compile=False)
            self.class_names = open('./newlabels.txt', 'r').readlines()
            with open('./new_data.json', encoding="utf8") as f:
                self.data = json.load(f)
        else:
            print("here")
            print(tp)
            self.model = load_model('./keras_model.h5', compile=False)
            self.class_names = open('./labels.txt','r').readlines()
            with open('./Data.json', encoding="utf8") as f:
                self.data = json.load(f)
    
    def predict_image(self,image_path):
    # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
        image = Image.open(image_path).convert('RGB')

    # Resize the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
        image_array = np.asarray(image)

    # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
        data[0] = normalized_image_array

    # Run the inference
        prediction = self.model.predict(data)
        index = np.argmax(prediction)

    # if prediction is less than 0.8 then its not a crop
        if(prediction[0][index]<0.8):
            return 38
        class_name = self.class_names[index].strip()
        
    # Extract class index from the class name
        arr = class_name.split(" ")
        class_index = int(arr[0])
        
        return self.data[class_index]
    

   