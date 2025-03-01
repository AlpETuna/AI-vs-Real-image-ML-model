import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def data_generator(df, batch_size=8):
    while True:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            X_batch = np.array(batch['image'].tolist()).astype('float32') / 255.0
            y_batch = np.array(batch['labels'])
            yield X_batch, y_batch

print('Loading dataset...')

label_load = np.load("AI_vs_real_image/labels.npy")  
image_load = np.load("AI_vs_real_image/data_wrangle.npy")


labels = pd.DataFrame(label_load, columns=['labels'])
images = pd.DataFrame({'image': list(image_load)})  


df = pd.concat([images, labels], axis=1, ignore_index=False)


max_val = df['labels'].max()
min_val = df['labels'].min()
range_val = max_val - min_val if max_val != min_val else 1
df['labels'] = (df['labels'] - min_val) / range_val 


train_df = df.sample(frac=0.75, random_state=4)
val_df = df.drop(train_df.index)

batch_size = 8
train_gen = data_generator(train_df, batch_size)
val_gen = data_generator(val_df, batch_size)

print("Data generator created successfully.")


model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid') 
])

print('Compiling model...')

model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])


model.summary()

print('Training model...')

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_df) // batch_size,
    validation_steps=len(val_df) // batch_size,
    epochs=20
)

model.save("image_classifier_model.h5")  
print("Model saved successfully!")


loss, accuracy = model.evaluate(val_gen, steps=len(val_df) // batch_size)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
