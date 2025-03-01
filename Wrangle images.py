import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import os

trainingPath = 'AI_vs_real_image/train.csv'
df = pd.read_csv(trainingPath)
df.drop(columns='Unnamed: 0', inplace=True)

image_data = []

for index, row in df.iterrows():
    path = os.path.join('AI_vs_real_image/', str(row['file_name']).replace("\ ",'/'))

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    IMAGE = keras.utils.load_img(path, color_mode="rgb", target_size=(224, 224))
    input_arr = keras.utils.img_to_array(IMAGE).astype('uint8')

    image_data.append(input_arr)



image_data = np.array(image_data)

np.save("data_wrangle.npy", image_data)