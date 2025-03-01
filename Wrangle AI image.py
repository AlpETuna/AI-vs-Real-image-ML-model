import numpy as np
import pandas as pd

image_data = np.load("AI_vs_real_image\data_wrangle.npy")
print("Loaded image data shape:", image_data.shape)

csv_df = pd.read_csv("AI_vs_real_image/train.csv")
csv_df.drop(columns=['Unnamed: 0'], inplace=True)


file_names = csv_df['file_name'].values 
sorted_indices = np.argsort(file_names) 

labels = csv_df['label'].values[sorted_indices]
print("Labels shape:", labels.shape)


labels = np.array(labels)
np.save("labels.npy", labels)

