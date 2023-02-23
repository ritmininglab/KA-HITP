
import tensorflow as tf
"""
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


import re
from tensorflow.keras.callbacks import Callback
from matplotlib.patches import Rectangle
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import backend as K
from skimage.transform import resize
import os

import scipy.io
import scipy.misc
import numpy as np
import math
import csv



"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
"""


class CustomCallback(Callback):
    def on_train_begin(self, logs={}):
        self.epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        if self.epochs % 2 == 1:
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} metric {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"] -
                    logs["lnow_loss"],
                logs["lnow_accuracy"])
                )


seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

adamlarge = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamverysmall = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
verbose = 0


"""
import csv
with open('eggs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
Texts = np.loadtxt(open('newToken.csv',"rb"),delimiter=",",skiprows=0)
"""



def loaddata1key(filename, foldername, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            """
            img_name, caption = line.split(",")
            img_name = line.strip()
            """
            caption = line.strip()
            if len(foldername) == 0: 
                caption_mapping.append(int(caption))
            elif foldername == 'nofoldername': 
                caption_mapping.append(caption)
            else:
                caption_mapping.append(foldername + caption)

        return caption_mapping

def loaddata2key(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            """
            img_name = cells[0].strip()
            """
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells), maxlength)):
                
                wordarray[i] = int(float(cells[i]))
            caption_mapping.append(wordarray)

        return caption_mapping


dictionary = []
dictionary.append('null')
with open('Dictionary.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        dictionary.append(row[-1])


def visualizedata(i, img2):
    if np.max(img2) > 1:
        img2 = img2.astype('uint8')
    plt.figure(figsize=(20, 20))
    plt.imshow(img2[i])
    plt.axis('off')
    plt.savefig('output/test'+str(i)+'.png', bbox_inches='tight')
    plt.show()
    plt.clf()


def translate(i, dictionary, prednum, printstr='pred'):
    T = prednum.shape[1]
    matchnow = prednum[i, 0:T]
    words = ''
    for t in range(T):
        wordidx = matchnow[t]
        if wordidx != 0:
            words += dictionary[wordidx]+' '
    print('Image '+str(i)+' '+printstr+': '+words)


N = 173200
h1 = 224  
w1 = 224  
Nbatch = 100  
T = 10-1
voc = 8443  

metatoken = loaddata1key('keywordmeta.csv', '', N)
imgpath = loaddata1key('keywordimgpath.csv', 'nofoldername', N)
tokendata = loaddata2key('keywordinput.csv', T+1, N)

mode = 0
if mode == 0:
    def structuredPermutation(metatoken, imgpath, tokendata, N):
        nimg = int(N/5)
        neworder = np.zeros((N)).astype(np.int32)
        for i in range(5):
            idxs = np.random.RandomState(seed=i).permutation(nimg)
            trueidxs = i + idxs*5
            neworder[i*nimg:(i+1)*nimg] = trueidxs
        metatoken2 = []
        imgpath2 = []
        tokendata2 = []
        for i in range(N):
            metatoken2.append(metatoken[neworder[i]])
            imgpath2.append(imgpath[neworder[i]])
            tokendata2.append(tokendata[neworder[i]])

        return metatoken2, imgpath2, tokendata2, neworder

    metatoken2, imgpath2, tokendata2, neworder = structuredPermutation(
        metatoken, imgpath, tokendata, N)

    debug = 1
    if debug == 1:
        N = 30000
        metatoken2 = metatoken2[0:N]
        imgpath2 = imgpath2[0:N]
        tokendata2 = tokendata2[0:N]
else:
    metatoken2 = metatoken
    imgpath2 = imgpath
    tokendata2 = tokendata


"""
x = tf.Variable(tf.random.uniform([5, 30], -1, 1))
s = tf.split(x, num_or_size_splits=3, axis=1)

x = tf.Variable(tf.random.uniform([1,4,4,3], -1, 1))
s = tf.compat.v1.nn.max_pool(x, ksize=(2,2), strides=(2,2), padding='VALID')

"""

from modelCap4 import attnTransformer as BFvgg
from pattern.text.en import singularize
from nltk.stem.snowball import SnowballStemmer
from utilVisualize import getFilter, attn2heatmap
from tensorflow.keras.applications.vgg16 import VGG16
from utilIO2 import get_train_batch_mask as get_train_batch_mask


mygenerator = get_train_batch_mask(imgpath2, tokendata2, metatoken2,
                                    Nbatch, h1, w1, T, voc)
numsteps = int(N/Nbatch)

mode = 0
if mode == 0:
    from utilIO2 import get_test_data
    imgs, token1, token2, masks, imgsraw = get_test_data(imgpath2, tokendata2, metatoken2,
                                                        5, h1, w1, T, voc, 0)
    for i in range(5):
        visualizedata(i, np.clip(imgs, 0, 1))
        visualizedata(i, imgsraw)
        translate(i, dictionary, token1, 'input')
        translate(i, dictionary, token2, 'targt')




dims2 = [250, 256, voc]  
mmt = 0.9
bn = True
hw = 7  
dimimg = 512  
"""
trainable = []
for i in range(16):
    trainable.append(True)
"""


ximg = Input(batch_shape=(Nbatch, h1, w1, 3), name='ximg')
xtxt = Input(batch_shape=(Nbatch, T), name='xtxt')



cnn = VGG16(include_top=False, weights="imagenet", input_tensor=ximg)
cnn.trainable = False

"""
def mylosslogitsparse(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True, axis=-1)
    return temp
"""


def myloss(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, axis=-1)
    return temp


def mylosslogit(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True, axis=-1)
    return temp


m0 = Model(inputs=[ximg, xtxt],
          outputs=BFvgg([ximg, xtxt], dims2, T, True, cnn, hw, dimimg, Nbatch))
m0.compile(loss={'lnow': mylosslogit,  
                        },
          loss_weights={'lnow': 1.0, },
          optimizer=adamsmall,
          metrics={'lnow': 'accuracy', })




mode = 1
if mode == 0:


    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=5,
        verbose=1,)

    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=5,
        verbose=1,)

    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=5,
        verbose=1,)


    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=15,
        verbose=1,)



    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=15,
        verbose=1,)


    
    m0.fit(mygenerator,
        steps_per_epoch=numsteps,
        epochs=15,
        verbose=1,)

