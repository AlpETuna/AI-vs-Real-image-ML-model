import numpy as np
import cv2  
from tensorflow import keras

model = keras.models.load_model("AI_vs_real_image\image_classifier_model.h5")


def predict_image(image_path):
   
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]  


    if prediction > 0.5:
        result = "AI-Generated"
    else:
        result = "Real"

    print(f"Prediction: {result} (Confidence: {prediction:.4f})")

# Example usage
predict_image("AI_vs_real_image\Image test.jpg")
