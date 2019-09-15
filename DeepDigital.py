import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import itertools 

np.random.seed(2)

def data_prep(raw):
    y = raw.label
    out_y = keras.utils.to_categorical(y,10)

    x = raw.values[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images,28,28,1)
    out_x = out_x/np.max(x)
    return out_x, out_y

def plot_conf_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], 
                horizontalalignment="center",
                color='white' if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#  plt.show()

def display_discrep(discrep_index, img_discrep, pred_discrep, obs_discrep):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            discrep = discrep_index[n]
            ax[row,col].imshow((img_discrep[discrep]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_discrep[discrep], obs_discrep[discrep]))
            n += 1
    fig.show()       
   
#Dataset Location 
data_dir = "/Users/daniellee/Documents/Kaggle_Datasets/HWR/digit-recognizer/"
test_data = pd.read_csv(data_dir+"test.csv")
train_data = pd.read_csv(data_dir+"train.csv")


# Initial Data Prep     
random_seed = 2

x, y = data_prep(train_data)
X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=random_seed)

# Creating the CNN model 
num_model = Sequential() 
num_model.add(Conv2D(32, kernel_size=(3,3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
num_model.add(Conv2D(32, kernel_size=(3,3),
                   activation='relu'))
num_model.add(MaxPool2D(pool_size=(2,2)))
num_model.add(Dropout(0.15))    
num_model.add(Flatten())
num_model.add(Dense(128,activation='relu'))
num_model.add(Dense(10,activation='softmax'))

#Optimizing 
num_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='sgd',
                  metrics=['accuracy'])

learning_rate = ReduceLROnPlateau(monitor="val_acc", patience=1, verbose=1, factor=0.40, min_lr=0.00001)

num_model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_val, Y_val), callbacks=[learning_rate]) 

#Creating confusino matrix     
Y_pred = num_model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)
plot_conf_matrix(confusion_mtx, classes=range(10))

#Checking causes of discrepancy    
discrep = (Y_pred_classes - Y_true !=0)

Y_pred_classes_discrep = Y_pred_classes[discrep]
Y_pred_discrep = Y_pred[discrep]
Y_true_discrep = Y_true[discrep]
X_val_discrep = X_val[discrep]

Y_pred_discrep_prob = np.max(Y_pred_discrep, axis=1)
true_prob_discrep = np.diagonal(np.take(Y_pred_discrep, Y_true_discrep, axis=1))
delta_pred_true_discrep = Y_pred_discrep_prob - true_prob_discrep
sorted_delta_discrep = np.argsort(delta_pred_true_discrep)
most_important_discrep = sorted_delta_discrep[-6:]

#display_discrep(most_important_discrep, X_val_discrep, Y_pred_classes_discrep, Y_true_discrep)

#Predicting test data 

test_x = test_data.values[:,0:]
test_num_images = test_data.shape[0]
test_out_x = test_x.reshape(test_num_images,28,28,1)
test_out_x = test_out_x/np.max(test_x)

preds = num_model.predict_classes(test_out_x)
#preds = np.argmax(preds, axis=1)
preds - pd.Series(preds,name="Label")

#submission = pd.concat([pd.Series(range(1, len(preds)+1, name="ImageId"), preds], axis=1) 
submission = pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "label":preds})
submission.to_csv("predicted_digits.csv", index=False)


