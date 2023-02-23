from __future__ import division
import numpy as np
from tensorflow.keras.utils import to_categorical


from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input


def get_img(paths, Nbatch,h1,w1, color_type=3, normalize=True):
    if color_type == 1:
        imgs = np.zeros((Nbatch,h1,w1)).astype('float32')
    elif color_type == 3:
        imgs = np.zeros((Nbatch,h1,w1,3)).astype('float32')
    for i in range(Nbatch):
        path = paths[i]
        if color_type == 1:
            img = imread(path)
        elif color_type == 3:
            img = imread(path)
        resized = imresize(img, (h1,w1))
        resized = (resized*255).astype(np.uint8)
        resized = preprocess_input(resized)
        imgs[i,:] = np.copy(resized)
        
    return imgs

def get_mask(datas1, Nbatch,T):
    masks = np.zeros((Nbatch,T)).astype('float32')
    for i in range(Nbatch): 
        masks[i, 0:min(datas1[i]-1,T-1)] = 1
    return masks

def get_mask3(datas1, Nbatch,T):
    masks = np.zeros((Nbatch,T)).astype('float32')
    for i in range(Nbatch): 
        validlength = np.sum(datas1[i]>0)
        validlength = min(validlength,T)
        masks[i, 0:validlength] = 8 / (validlength)
    return masks

def get_mask4(datas1, Nbatch,T):
    masks = np.zeros((Nbatch,T)).astype('float32')
    for i in range(Nbatch): 
        validlength = np.sum(datas1[i]>0)
        validlength = min(validlength,T)
        masks[i, 0:validlength] = 8 / (validlength)
    
    masksdiv = np.ones((Nbatch,T)).astype('float32')
    """
    """
    masksdiv = masksdiv + (datas1==3).astype('float32')
    masks = masks/masksdiv
    return masks

def get_token(datas, Nbatch,T,voc):
    tokens = np.zeros((Nbatch,T)).astype('int32')
    for i in range(Nbatch):
        tokens[i,:] = np.copy(datas[i])
    return tokens

def get_img2(paths, Nbatch,h1,w1, color_type=3, normalize=True):
    if color_type == 1:
        imgs = np.zeros((Nbatch,h1,w1)).astype('float32')
    elif color_type == 3:
        imgs = np.zeros((Nbatch,h1,w1,3)).astype('float32')
    for i in range(Nbatch):
        path = paths[i]
        if color_type == 1:
            img = imread(path)
        elif color_type == 3:
            img = imread(path)
        resized = imresize(img, (h1,w1))
        imgs[i,:] = np.copy(resized)
        
    return imgs

def get_train_batch_mask(imgpath,insertinput,inserttarget,Nbatch,h1,w1,T,voc):
    while 1:
        for i in range(0, len(imgpath), Nbatch):
            imgs = get_img(imgpath[i:i+Nbatch], Nbatch,h1,w1,3,True)
            token1 = get_token(insertinput[i:i+Nbatch], Nbatch,T,voc)
            token2 = get_token(inserttarget[i:i+Nbatch], Nbatch,T,voc)
            masks = get_mask4(token2, Nbatch,T)
            yield([imgs,token1], {'lnow': token2}, masks)
def get_test_data(imgpath,insertinput,inserttarget,Nbatch,h1,w1,T,voc, startidx):
    i = startidx
    imgs = get_img(imgpath[i:i+Nbatch], Nbatch,h1,w1,3,True)
    token1 = get_token(insertinput[i:i+Nbatch], Nbatch,T,voc)
    token2 = get_token(inserttarget[i:i+Nbatch], Nbatch,T,voc)
    masks = get_mask4(token2, Nbatch,T)
    imgsraw = get_img2(imgpath[i:i+Nbatch], Nbatch,h1,w1,3,True)
    return imgs,token1, token2, masks, imgsraw

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

def deBN_noVar(bnweights, convweights):
    epsilon = 0.001 
    
    gamma = bnweights[0]
    beta = bnweights[1]
    mamean = bnweights[2]
    mavar = bnweights[3]
    conv = convweights[0]
    bias = convweights[1]
    temp = gamma / np.sqrt(mavar+epsilon)
    conv2 = conv * temp
    bias2 = (bias-mamean) * temp + beta
    return [conv2, bias2]
def modeldeBN_noVar(m, m3):
    layernames = ['b1b','b2b','b3b','b4b','b5b']
    layernamesbn = ['d1','d2','d3','d4','d5']
    for i in range(len(layernames)):
        idx1 = getlayeridx(m, layernames[i])
        idx2 = getlayeridx(m3, layernames[i])
        idxbn = getlayeridx(m, layernamesbn[i])
        convweights = m.layers[idx1].get_weights()
        bnweights = m.layers[idxbn].get_weights()
        newweight = deBN_noVar(bnweights, convweights)
        m3.layers[idx2].set_weights(newweight)
        
    layernames2 = ['b1a','b2a','b3a','b4a','b5a',
                   'embed','lrnn','embed2']
    for layername in layernames2:
        newweight = getlayerweights(m, layername)
        idx2 = getlayeridx(m3, layername)
        m3.layers[idx2].set_weights(newweight)
    return m3
