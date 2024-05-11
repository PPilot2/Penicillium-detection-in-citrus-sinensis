# 1. Install Dependencies and Setup
import tensorflow, matplotlib
import statistics
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import prettytable as pt
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from prettytable import PrettyTable
from tkinter import filedialog
import tkinter as tk
import json
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
# 2. Remove dodgy images
model = Sequential()
accuracy = np.array([])
def train():
    global model, hist, accuracy
    data_dir = 'data' 
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)
    # 3. Load Data
    
    data = tf.keras.utils.image_dataset_from_directory('data')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    # 4. Scale Data
    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()
    # 5. Split Data
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    # 6. Build Deep Learning Model
    # model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # model.summary()
    # 7. Train
    global logdir, tensorboard_callback, hist
    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # print(train)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
    # 8. Plot Performance
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    # plt.show()
    fig = plt.figure()
    accuracy = np.array(hist.history['accuracy'])
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    dictForData = {"accuracy": hist.history['accuracy'], "val_accuracy": hist.history['val_accuracy'], 'loss': hist.history['loss'], "val_loss": hist.history['val_loss']}
    with open("data.json", "w") as f:
        dataDict = json.dump(dictForData, f)
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    myTable = PrettyTable(["Epoch Number", "Loss", "Accuracy"])
    myTable.add_column("Epoch Number", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    myTable.add_column("Loss", hist.history['loss'])
    myTable.add_column("Accuracy", hist.history['accuracy'])
    # print(myTable)
    # plt.show()
    # 9. Evaluate
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    model.save(os.path.join('models','imageclassifier.h5'))
    new_model = load_model(os.path.join('models','imageclassifier.h5'), compile=False)
    # print(pre.result(), re.result(), acc.result())
# train()
# 10. Test
def upload():
    f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png')] 
    path = filedialog.askopenfilename(filetypes=f_types)
    testImg = cv2.imread(path)
    test(testImg)

def test(img):
    global model
    plt.imshow(img)
    # plt.show()
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    # plt.show()
    yhat = model.predict(np.expand_dims(resize/255, 0))
    yhat
    if yhat.all() > 0.5: 
        result = f'Predicted class is Negative'
    else:
        result = f'Predicted class is Positive'
    resultLabel.config(text="")
    resultLabel.config(text=result)
    # 11. Save the Model
    model.save(os.path.join('models','imageclassifier.h5'))
    new_model = load_model(os.path.join('models','imageclassifier.h5'), compile=False)
    new_model
    new_model.predict(np.expand_dims(resize/255, 0))

#data results
with open('data.json', 'r') as f:
    accuracy = json.load(f)['accuracy']
avgAccuracy = 0
for i in accuracy:
    avgAccuracy += int(i)
avgAccuracy = avgAccuracy/len(accuracy)

# Calculate the mean
mean = np.mean(accuracy)

# Calculate the variance
variance = np.var(accuracy)

# Calculate the standard deviation
std_deviation = np.std(accuracy)

# Store the calculated statistics
output = "Mean: " + str(round(mean, 3)) + ". Variance: " + str(round(variance, 3)) + ". Standard Deviation: " + str(round(std_deviation, 3)) + ". Average accuracy: " + str(round(avgAccuracy, 3))

def openStats():
    statsLabel.config(text=output)

window = tk.Tk()
titleLabel = tk.Label(text="Enter an image", font=("Arial", 25))
titleLabel.grid(row=0, column=0)

uploadButton = tk.Button(text="Upload", command=upload, font=("Arial", 15))
uploadButton.grid(row=1, column=0)

resultLabel = tk.Label(text="", font=("Arial", 15))
resultLabel.grid(row=2, column=0)

openStatsButton = tk.Button(text="Show data", command=openStats, font=("Arial", 15))
openStatsButton.grid(row=3, column=0)

statsLabel = tk.Label(text="", font=("Arial", 15))
statsLabel.grid(row=4, column=0)

window.mainloop()