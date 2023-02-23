import os

import scipy.io
import scipy.misc
import numpy as np
from numpy import expand_dims
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
from tensorflow.keras.callbacks import Callback
from config import p1,p2,p3,p4
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import SimpleRNN as RNN


class CustomCallback(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 2 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} metric {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

adamlarge = tf.keras.optimizers.Adam(
    learning_rate=0.0025, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
adamsmall = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
verbose = 0




def loaddata1(filename, foldername, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            caption_mapping.append(foldername + line)

        return caption_mapping

def loaddata2(filename, maxlength, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            cells = line.split(",")
            wordarray = np.zeros((maxlength,)).astype(np.int32)
            for i in range(0, min(len(cells), maxlength)):
                wordarray[i] = int(float(cells[i]))
            caption_mapping.append(wordarray)

        return caption_mapping

def loaddatascalar(filename, N):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = []
        for idx in range(N):
            line = caption_data[idx]
            line = line.rstrip("\n")
            caption_mapping.append(int(line))

        return caption_mapping


idxNI = 3

import pickle
testimgs, testimgsraw, testimgfeature, dictionary \
    = pickle.load(open('test/sample.pkl',"rb"))

def visualizedata(i, img2):
    if np.min(img2)>1:
        img2 = img2.astype('uint8')
    plt.figure(figsize=(20, 20))
    plt.imshow(img2[i])
    plt.axis('off')
    plt.show()
    plt.clf()


def translate(i, dictionary, prednum, printstr='pred'):
    T = prednum.shape[1]
    matchnow = prednum[i,0:T]
    words = ''
    for t in range(T):
        wordidx = matchnow[t]
        if wordidx !=0:
            words += dictionary[wordidx]+' '
    print('Image '+str(i)+' '+printstr+': '+words)
    

N = 186187 
h1 = 224
w1 = 224
Nbatch = 50 
T = 25-1
Tk = 8
voc = 8495 



from utilIOkey import get_train_insertion as get_train
from utilIOkey import get_test_insertion as get_test

from tensorflow.keras.applications.vgg16 import VGG16
xcnn = Input(shape=(h1,w1, 3), name='xcnn')
cnn = VGG16(include_top=False, weights="imagenet", input_tensor=xcnn)
cnn.trainable = False

Nleap = 10 
capa = 8
hw = 7  
dimimg = 512
Nstep = int(40400/(Nleap*5))





dims = [300, 64, voc, capa]
priors = []
for i in range(15):
    priors.append([0.,1.,0.,1.,])
priors2 = []
for i in range(5):
    priors2.append([5.,1.,])
kldiv = 40000*20*1000
bnparam = [[priors, priors2],kldiv,[True,True]]

dimsk = [300, 64, voc, capa]
priorsk = []
for i in range(9):
    priorsk.append([0.,1.,0.,1.,])
priors2k = []
for i in range(3):
    priors2k.append([5.,1.,])
kldivk = 40000*10*1000
bnparamk = [[priorsk, priors2k],kldivk,[True,True]]



ximg = Input(batch_shape=(Nbatch, h1,w1, 3), name='ximg')
xtxt = Input(batch_shape=(Nbatch, T), name='xtxt')



def myloss(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, axis=-1)
    return temp
def mylosslogit(y_true, y_pred):
    temp = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True, axis=-1)
    return temp

ximg = Input(shape=(hw, hw, dimimg), name='ximg')
xtxt = Input(shape=(T), name='xtxt', dtype='int32')
xtxtp = Input(shape=(T,T), name='xtxtp', dtype='float32')
xtxtkb = Input(shape=(capa,1), name='xtxtkb') 
xtxtq = Input(shape=(capa+T), name='xtxtq', dtype='int32') 
xtxtk = Input(shape=(Tk), name='xtxtk', dtype='int32') 
xtxtkp = Input(shape=(Tk,Tk), name='xtxtkp', dtype='float32') 
xtxtkq = Input(shape=(capa+Tk), name='xtxtkq', dtype='int32') 

from mCap import AtnInsertTransformer as BFvgg

m1 = Model(inputs=[ximg, xtxt, xtxtkb,xtxtp,xtxtq,xtxtk,xtxtkp],
          outputs=BFvgg([ximg, xtxt, xtxtkb,xtxtp,xtxtq,xtxtk,xtxtkp],
                        dims, T,Tk, hw, dimimg, bnparam))
m1.compile(loss={'lnow': mylosslogit,  
                        },
          loss_weights={'lnow': 1.0, },
          optimizer=adam,
          metrics={'lnow': 'accuracy', })


m1.load_weights(p2)




from mCap import AtnKeywordTransformer as BFvggk
m0 = Model(inputs=[ximg, xtxtk, xtxtkb,xtxtkp,xtxtkq],
          outputs=BFvggk([ximg, xtxtk, xtxtkb,xtxtkp,xtxtkq], dimsk, Tk, hw, dimimg, bnparamk))
m0.compile(loss={'lnow': mylosslogit,  
                        },
          loss_weights={'lnow': 1.0, },
          optimizer=adam,
          metrics={'lnow': 'accuracy', })

m0.load_weights(p1)





def mcpred(m, inputs, mc, i):
    preds = m.predict(inputs)
    mode1 = preds[i]
    for itr in range(1,mc):
        preds = m.predict(inputs)
        mode1 = mode1+ preds[i]
    mode1 = mode1/mc
    return [mode1,]

def topKcandidate3update(candidates5,scores5,temp2,temp2score, beam):
    if len(candidates5)<beam:
        candidates5.append(temp2)
        scores5.append(temp2score)
    else:
        scores3np = np.asarray(scores5)
        minscore = np.min(scores3np)
        if temp2score>minscore: 
            minidx = np.where(scores3np==minscore)[0][0]
            candidates5[minidx] = temp2
            scores5[minidx] = temp2score
    return candidates5, scores5

def topKcandidate3rank(candidates5,scores5,beam):
    candidates4 = []
    scores4 = []
    order = np.argsort(scores5)
    for i in range(len(order)):
        pos = order[len(order) - i-1]
        candidates4.append(candidates5[pos])
        scores4.append(scores5[pos])
    return candidates4, scores4

def translatebeam1(i, dictionary, candidates5, scores5):
    T = candidates5[0].shape[1]
    beam = len(candidates5)
    for i in range(beam):
        matchnow = candidates5[i]
        words = ''
        for t in range(T):
            wordidx = matchnow[0,t]
            if wordidx >3:
                words += dictionary[wordidx]+' '
        print(words)


dictwordidx = {}
for j in range(len(dictionary)):
    word = dictionary[j]
    dictwordidx[word] = j
def userKeywords2token(userstr, Tk, dictwordidx):
    useranno = userstr.split(",")
    tokens = np.zeros((1,Tk)).astype('int32')
    for i in range(len(useranno)):
        pos = i
        word = useranno[i]
        tokens[0,pos] = dictwordidx[word]
    tokens[0,len(useranno)] = 2 
    return tokens

def checkNInew(predtoken, caption0, key1, t, idxNI):
    if predtoken==idxNI:
        newcaption = np.copy(caption0).astype('int32')
        newcaption[0,t+1] = key1[0,0]
        newkey = np.copy(key1).astype('int32')
        newkey[0,0:-1] = key1[0,1:]
        newkey[0,-1] = 0
    else:
        newcaption = np.copy(caption0).astype('int32')
        newcaption[0,t+1] = predtoken
        newkey = np.copy(key1).astype('int32')
    iscomplete = False
    if predtoken==idxNI and newkey[0,0] == 0:
        iscomplete = True
    return newcaption,newkey, iscomplete







textbox = 0
realuser = 2 
if textbox==0:
    import tkinter as tk
    from PIL import ImageTk, Image
    from imageio import imread,imwrite
    import cv2
    
    
    imgpath2 = 'test/0.gif'
    
    wh1=224
    wh2 = int(wh1/2)
    
    res2 = testimgsraw[1]
    imwrite(imgpath2,res2)
    
    userstr = ""  
    def getwords():
        global userstr
        userstr = entry.get()
        
        
        
    root = tk.Tk()
    root.title("Specify keywords")
    canvas = tk.Canvas(root, width=wh1,height=wh1, bd=0, highlightthickness=0)
    
    img = Image.open(imgpath2)
    photo = ImageTk.PhotoImage(img)
    
    canvas.create_image(wh2,wh2,image=photo)
    canvas.pack()
    entry = tk.Entry(root, insertbackground='white',highlightthickness=2)
    entry.pack()
    
    button_calc = tk.Button(root, text="Specify keywords and submit", command=getwords)
    button_calc.pack()
    
    root.mainloop()

usertoken = userKeywords2token(userstr, Tk, dictwordidx)




import itertools
def usertokenPermutate(usertoken, Tk):
    userlist = []
    for i in range(usertoken.shape[1]):
        token = usertoken[0,i]
        if token>idxNI:
            userlist.append(token)
    permulist = list(itertools.permutations(userlist))
    userlist2 = []
    for permu in permulist:
        tokens = np.zeros((1,Tk)).astype('int32')
        tokens[0,0] = 1 
        for i in range(len(permu)):
            tokens[0,i+1] = permu[i]
        tokens[0,len(permu)+1] = 2
        userlist2.append(tokens)
    return userlist2
    
userlist2 = usertokenPermutate(usertoken, Tk)
from scipy.special import softmax
def mcpred2(m, inputs, mc, i):
    preds = m.predict(inputs)
    mode1 = softmax(preds[i],-1)
    for itr in range(1,mc):
        preds = m.predict(inputs)
        mode1 = mode1+ softmax(preds[i],-1)
    mode1 = mode1/mc
    mode2 = np.log(mode1+1e-5)
    return [mode2,]
mc = 16


i = 1
numvalid = np.sum(userlist2[0]>0)
img2feature = testimgfeature[i:i+1, :]
img2raw = testimgsraw[i:i+1, :]
posembed = np.expand_dims(np.eye(Tk), axis=0)
qpseu0 = np.ones((1,capa), dtype='int32')
kbpseu = np.zeros((1,capa,1), dtype='float32')
ll = np.zeros((len(userlist2),))
for i in range(len(userlist2)):
    temp = userlist2[i]
    qpseu = np.concatenate((qpseu0, temp), axis=-1)
    preds = mcpred2(m0, [img2feature, temp, kbpseu, posembed, qpseu], mc,0)
    logit = preds[0][0, 0:numvalid]
    temp2 = 0
    for j in range(numvalid-1):
        word = temp[0,j+1]
        temp2 = temp2 + logit[j,word]
    ll[i] = temp2

optimalidx = np.argmax(ll)
optimal = np.zeros(usertoken.shape, dtype='int32')
optimal[0,0:Tk-1] = np.copy(userlist2[optimalidx][0,1:])
usertoken = optimal







mc = 16
Nbatch2 = 1
beam = 3
beamkey = 1
beamshow = 3
candidatesall3 = [] 
scoresall3 = []


startingkey = usertoken
startingscore = 0

candidates1 = np.zeros((beam, T)).astype(np.int32)
candidates2 = np.zeros((beam*beam, T)).astype(np.int32)
candidates1pos = np.zeros((beam, Tk)).astype(np.int32)
candidates2pos = np.zeros((beam*beam, Tk)).astype(np.int32)

scores1 = -1000*np.ones((beam,))
scores2 = -1000*np.ones((beam*beam))
candidates5 = [] 
scores5 = []



i = 1
if beam > 0:
    img2feature = testimgfeature[i:i+1, :]
    img2raw = testimgsraw[i:i+1, :]
    visualizedata(0, img2raw)

    key1 = startingkey
    candidates1[:, 0] = 1
    t = 0
    temp = candidates1[0:1]
    posembed = np.expand_dims(np.eye(T), axis=0)
    posembed2 = np.expand_dims(np.eye(Tk), axis=0)
    kbpseu = np.zeros((Nbatch2,capa,1), dtype='float32')
    querypseu0 = np.ones((Nbatch2,capa), dtype='int32')
    querypseu = np.concatenate((querypseu0, temp), axis=-1)
    
    preds = mcpred(m1,[img2feature,temp,kbpseu,posembed,querypseu,key1,posembed2], mc,0)
    logit = preds[0][0, t]
    logit = logit - np.max(logit)
    lls = logit - np.log( np.sum(np.exp(logit)))
    ranked = np.argsort(-lls)
    for j in range(beam):
        word = ranked[j]
        newcaption,newkey,iscomplete = checkNInew(word, temp, key1, t, idxNI)
        candidates1[j:j+1] = newcaption
        candidates1pos[j:j+1] = newkey
        scores1[j] = lls[ranked[j]] + startingscore
    
    validcandi1 = beam
    for t in range(1, T-1):
        validcandi2 = np.zeros((beam,)).astype('int8')
        scores2 = -1000*np.ones((beam*beam))
        for k in range(min(beam,validcandi1)):
            temp = candidates1[k:k+1]
            querypseu = np.concatenate((querypseu0, temp), axis=-1)
            key1 = candidates1pos[k:k+1]
            preds = mcpred(m1,[img2feature,temp,kbpseu,posembed,querypseu,key1,posembed2], mc,0)
            logit = preds[0][0, t]
            logit = logit - np.max(logit)
            lls = logit - np.log( np.sum(np.exp(logit)))
            ranked = np.argsort(-lls)
            for j in range(beam):
                word = ranked[j]
                newcaption,newkey,iscomplete = checkNInew(word, temp, key1, t, idxNI)
                temp2 = newcaption
                temp2key = newkey
                temp2score = scores1[k] + lls[ranked[j]]
                
                if iscomplete:
                    candidates5,scores5 = topKcandidate3update(candidates5,scores5,temp2,temp2score, beam*2)
                else: 
                    idxcandi = np.sum(validcandi2)
                    candidates2[idxcandi:idxcandi+1] = temp2
                    candidates2pos[idxcandi:idxcandi+1] = temp2key
                    scores2[idxcandi] = temp2score
                    validcandi2[k] = 1
                
                
        validcandi1 = np.sum(validcandi2>0)
        if validcandi1==0: 
            break
        
        worstcomplete = -1000
        if len(scores5)>beam: 
            worstcomplete = min(scores5)
        bestincomplete = np.max(scores2)
        if worstcomplete > bestincomplete:
            break
        
        ranked = np.argsort(-scores2)
        for k in range(min(beam, np.sum(validcandi2))):
            candidates1[k] = candidates2[ranked[k]]
            candidates1pos[k] = candidates2pos[ranked[k]]
            scores1[k] = scores2[ranked[k]]
            

candidates6, scores6 = topKcandidate3rank(candidates5,scores5,beam)

cut = min(3,beamshow)
print('')
print('Refined candidate captions:')
translatebeam1(i, dictionary, candidates6[:cut], scores6[:cut])
print('')





"""
def preparePriors5(m, reducevalue, lnowreducevalue, lW,layernameZ):
    priors = []
    for idx in range(len(layernameZ)):        
        idxW = getlayeridx(m, lW[idx]+'a')
        idxZ = getlayeridx(m, layernameZ[idx]+'a')
        wweights = m.layers[idxW].get_weights()
        abzweights = m.layers[idxZ].get_weights()
        vpriora = np.log(np.exp(abzweights[0])+1)
        vpriorb = np.log(np.exp(abzweights[1])+1)
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        bpriormu = wweights[2]
        bpriorvar = np.exp(wweights[3]-reducevalue)
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
    
    idxW = getlayeridx(m, 'lnowa')
    wweights = m.layers[idxW].get_weights()
    wpriormu = wweights[0]
    wpriorvar = np.exp(wweights[1]-lnowreducevalue)
    bpriormu = wweights[2]
    bpriorvar = np.exp(wweights[3]-lnowreducevalue)
    priors.append([1,1, wpriormu, wpriorvar, bpriormu, bpriorvar])
    return priors
"""
def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index
def getlayerweights(model, layername):
    idx = getlayeridx(model, layername)
    return model.layers[idx].get_weights() 
def preparePrior(m,reducevalue):
    lW = ['embed','embed2','embed3a','fd1','embedk','embedk2','fd2','f4b1','f6b','f6b1','dense0','lnow']
    layernameZ = ['embed3','fd1b','fd2b']
    priors = []
    priors2 = []
    for idx in range(len(layernameZ)):        
        idxZ = getlayeridx(m, layernameZ[idx])
        abzweights = m.layers[idxZ].get_weights()
        vpriora = np.log(np.exp(abzweights[0])+1)
        vpriorb = np.log(np.exp(abzweights[1])+1)
        priors2.append([vpriora, vpriorb])

    for idx in range(len(lW)):      
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        if idx==2:
            wpriormu = wweights[0]
            wpriorvar = np.exp(wweights[1]-reducevalue)
            priors.append([wpriormu, wpriorvar,])
        else:
            wpriormu = wweights[0]
            wpriorvar = np.exp(wweights[1]-reducevalue)
            bpriormu = wweights[2]
            bpriorvar = np.exp(wweights[3]-reducevalue)
            priors.append([wpriormu, wpriorvar, bpriormu, bpriorvar])

    return priors,priors2



priorsnew, priors2new = preparePrior(m1, 0)



bnparam2 = [[priorsnew, priors2new],kldiv,[True,True]]
m3 = Model(inputs=[ximg, xtxt, xtxtkb,xtxtp,xtxtq,xtxtk,xtxtkp],
          outputs=BFvgg([ximg, xtxt, xtxtkb,xtxtp,xtxtq,xtxtk,xtxtkp],
                        dims, T,Tk, hw, dimimg, bnparam2))
m3.compile(loss={'lnow': mylosslogit,  
                        },
          loss_weights={'lnow': 1.0, },
          optimizer=adamsmall,
          metrics={'lnow': 'accuracy', })


    

def preparePriork(m,reducevalue):
    lW = ['embed','embed2','embed3a','fd1','f5b','f5b1','dense0','lnow']
    layernameZ = ['embed3','fd1b']
    priorsk = []
    priors2k = []
    for idx in range(len(layernameZ)):        
        idxZ = getlayeridx(m, layernameZ[idx])
        abzweights = m.layers[idxZ].get_weights()
        vpriora = np.log(np.exp(abzweights[0])+1)
        vpriorb = np.log(np.exp(abzweights[1])+1)
        priors2k.append([vpriora, vpriorb])

    for idx in range(len(lW)):      
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        if idx==2:
            wpriormu = wweights[0]
            wpriorvar = np.exp(wweights[1]-reducevalue)
            priorsk.append([wpriormu, wpriorvar,])
        else:
            wpriormu = wweights[0]
            wpriorvar = np.exp(wweights[1]-reducevalue)
            bpriormu = wweights[2]
            bpriorvar = np.exp(wweights[3]-reducevalue)
            priorsk.append([wpriormu, wpriorvar, bpriormu, bpriorvar])

    return priorsk,priors2k

priorsnewk, priors2newk = preparePriork(m0, 0)


bnparam2 = [[priorsnewk, priors2newk],kldivk,[True,True]]
m2 = Model(inputs=[ximg, xtxtk, xtxtkb,xtxtkp,xtxtkq],
          outputs=BFvggk([ximg, xtxtk, xtxtkb,xtxtkp,xtxtkq], dimsk, Tk, hw, dimimg, bnparam2))
m2.compile(loss={'lnow': mylosslogit,  
                        },
          loss_weights={'lnow': 1.0, },
          optimizer=adamsmall,
          metrics={'lnow': 'accuracy', })

rtoken1 = np.ones(usertoken.shape, dtype='int32')
rtoken1[0,1:] =  usertoken[0,0:-1]

mode = 1
if mode == 1:
    rimgs = img2feature
    rkbpseu = np.zeros((1,capa,1))
    rposembed = np.expand_dims(np.eye(Tk),0)
    rqpseu = np.ones((1,capa+Tk))
    rqpseu[0,capa:] = np.copy(rtoken1[0])
    rtoken2 = np.zeros(rtoken1.shape)
    rtoken2[0,0:-1] = rtoken1[0,1:]
    rmasks = (rtoken2>0).astype('float32')
    
    
m2.fit([rimgs,rtoken1,rkbpseu,rposembed,rqpseu],
    {'lnow': rtoken2},
    sample_weight = rmasks,
    epochs=50,
    verbose=0,)
    
    
    
    



rimgs = img2feature
rtoken1 = candidates6[0]
rkbpseu = np.zeros((1,capa,1))
rposembed = np.expand_dims(np.eye(T),0)
rqpseu = np.ones((1,capa+T))
rqpseu[0,capa:] = np.copy(rtoken1[0])
rkey1 = np.zeros((1,Tk)).astype('int32')
rkey1[0,0] = 2
rposembed2 = np.expand_dims(np.eye(Tk),0)
rtoken2 = np.zeros(rtoken1.shape)
rtoken2[0,0:-1] = rtoken1[0,1:]
rmasks = (rtoken2>0).astype('float32')



m3.fit([rimgs,rtoken1,rkbpseu,rposembed,rqpseu,rkey1,rposembed2],
    {'lnow': rtoken2},
    sample_weight = rmasks,
    epochs=50,
    verbose=0,)






m2.load_weights(p3)

mc = 16

beam = 3
beamshow = 3
candidates1 = np.zeros((beam, Tk)).astype(np.int32)
candidates2 = np.zeros((beam*beam, Tk)).astype(np.int32)
scores1 = -1000*np.ones((beam,))
scores2 = -1000*np.zeros((beam*beam))
candidates3 = [] 
scores3 = []
   
i=2
if beam > 0:
    img2 = testimgfeature[i:i+1, :]
    img2raw = testimgsraw[i:i+1, :]
    visualizedata(0, img2raw)
    
    candidates1[:, 0] = 1
    t = 0
    temp = candidates1[0:1]
    posembed = np.expand_dims(np.eye(Tk), axis=0)
    qpseu0 = np.ones((1,capa), dtype='int32')
    qpseu = np.concatenate((qpseu0, temp), axis=-1)
    kbpseu = np.zeros((1,capa,1), dtype='float32')
    preds = mcpred(m2, [img2, temp, kbpseu, posembed, qpseu], mc,0)
    logit = preds[0][0, t]
    logit = logit - np.max(logit)
    lls = logit - np.log( np.sum(np.exp(logit)))
    ranked = np.argsort(-lls)
    for j in range(beam):
        candidates1[j, t+1] = ranked[j]
        scores1[j] = lls[ranked[j]]
    
    validcandi1 = beam
    for t in range(1, Tk-1):
        validcandi2 = np.zeros((beam,)).astype('int8')
        scores2 = -1000*np.ones((beam*beam))
        for k in range(min(beam,validcandi1)):
            temp = candidates1[k:k+1]
            qpseu = np.concatenate((qpseu0, temp), axis=-1)
            preds = mcpred(m2, [img2, temp, kbpseu, posembed, qpseu], mc,0)
            logit = preds[0][0, t]
            logit = logit - np.max(logit)
            lls = logit - np.log( np.sum(np.exp(logit)))
            ranked = np.argsort(-lls)
            for j in range(beam):
                word = ranked[j]
                temp2 = np.copy(temp)
                temp2[0,t+1] = word
                temp2score = scores1[k] + lls[ranked[j]]
                
                if word==2: 
                    candidates3,scores3 = topKcandidate3update(candidates3,scores3,temp2,temp2score, beam*2)
                else: 
                    idxcandi = np.sum(validcandi2)
                    candidates2[idxcandi] = temp2
                    scores2[idxcandi] = temp2score
                    validcandi2[k] = validcandi2[k] + 1
        
        validcandi1 = np.sum(validcandi2>0)
        if validcandi1==0: 
            break
        ranked = np.argsort(-scores2)
        for k in range(min(beam, np.sum(validcandi2))):
            candidates1[k] = candidates2[ranked[k]]
            scores1[k] = scores2[ranked[k]]

candidates4, scores4 = topKcandidate3rank(candidates3,scores3,beam)
cut = min(5,beamshow)

print('')
print('Automated prediction of key concepts:')
translatebeam1(i, dictionary, candidates4[:cut], scores4[:cut])


usertoken = np.zeros(candidates4[0].shape)
usertoken[0,0:-1] =  candidates4[0][0,1:]


m3.load_weights(p4)



mc = 16
Nbatch2 = 1
beam = 3
beamkey = 1
beamshow = 3
candidatesall3 = [] 
scoresall3 = []


startingkey = usertoken
startingscore = 0

candidates1 = np.zeros((beam, T)).astype(np.int32)
candidates2 = np.zeros((beam*beam, T)).astype(np.int32)
candidates1pos = np.zeros((beam, Tk)).astype(np.int32)
candidates2pos = np.zeros((beam*beam, Tk)).astype(np.int32)

scores1 = -1000*np.ones((beam,))
scores2 = -1000*np.ones((beam*beam))
candidates5 = [] 
scores5 = []



i = 2
if beam > 0:
    img2feature = testimgfeature[i:i+1, :]
    img2raw = testimgsraw[i:i+1, :]

    key1 = startingkey
    candidates1[:, 0] = 1
    t = 0
    temp = candidates1[0:1]
    posembed = np.expand_dims(np.eye(T), axis=0)
    posembed2 = np.expand_dims(np.eye(Tk), axis=0)
    kbpseu = np.zeros((Nbatch2,capa,1), dtype='float32')
    querypseu0 = np.ones((Nbatch2,capa), dtype='int32')
    querypseu = np.concatenate((querypseu0, temp), axis=-1)
    
    preds = mcpred(m3,[img2feature,temp,kbpseu,posembed,querypseu,key1,posembed2], mc,0)
    logit = preds[0][0, t]
    logit = logit - np.max(logit)
    lls = logit - np.log( np.sum(np.exp(logit)))
    ranked = np.argsort(-lls)
    for j in range(beam):
        word = ranked[j]
        newcaption,newkey,iscomplete = checkNInew(word, temp, key1, t, idxNI)
        candidates1[j:j+1] = newcaption
        candidates1pos[j:j+1] = newkey
        scores1[j] = lls[ranked[j]] + startingscore
    
    validcandi1 = beam
    for t in range(1, T-1):
        validcandi2 = np.zeros((beam,)).astype('int8')
        scores2 = -1000*np.ones((beam*beam))
        for k in range(min(beam,validcandi1)):
            temp = candidates1[k:k+1]
            querypseu = np.concatenate((querypseu0, temp), axis=-1)
            key1 = candidates1pos[k:k+1]
            preds = mcpred(m3,[img2feature,temp,kbpseu,posembed,querypseu,key1,posembed2], mc,0)
            logit = preds[0][0, t]
            logit = logit - np.max(logit)
            lls = logit - np.log( np.sum(np.exp(logit)))
            ranked = np.argsort(-lls)
            for j in range(beam):
                word = ranked[j]
                newcaption,newkey,iscomplete = checkNInew(word, temp, key1, t, idxNI)
                temp2 = newcaption
                temp2key = newkey
                temp2score = scores1[k] + lls[ranked[j]]
                
                if iscomplete:
                    candidates5,scores5 = topKcandidate3update(candidates5,scores5,temp2,temp2score, beam*2)
                else: 
                    idxcandi = np.sum(validcandi2)
                    candidates2[idxcandi:idxcandi+1] = temp2
                    candidates2pos[idxcandi:idxcandi+1] = temp2key
                    scores2[idxcandi] = temp2score
                    validcandi2[k] = 1
                
                
        validcandi1 = np.sum(validcandi2>0)
        if validcandi1==0: 
            break
        
        worstcomplete = -1000
        if len(scores5)>beam: 
            worstcomplete = min(scores5)
        bestincomplete = np.max(scores2)
        if worstcomplete > bestincomplete:
            break
        
        ranked = np.argsort(-scores2)
        for k in range(min(beam, np.sum(validcandi2))):
            candidates1[k] = candidates2[ranked[k]]
            candidates1pos[k] = candidates2pos[ranked[k]]
            scores1[k] = scores2[ranked[k]]
            

candidates6, scores6 = topKcandidate3rank(candidates5,scores5,beam)

cut = min(3,beamshow)
print('')
print('Refined candidate captions:')
translatebeam1(i, dictionary, candidates6[:cut], scores6[:cut])
print('')
