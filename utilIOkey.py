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

    masksdiv = masksdiv + (datas1==3).astype('float32')
    masks = masks/masksdiv
    return masks


def get_mask_seqinsert(insertpos, Nbatch,T):
    masks = np.zeros((Nbatch,T)).astype('float32')
    for i in range(Nbatch): 
        masks[i, insertpos[i]] = 1
    return masks

def get_token(datas, Nbatch,T,voc):
    tokens = np.zeros((Nbatch,T)).astype('int32')
    for i in range(Nbatch):
        tokens[i,:] = np.copy(datas[i])
    return tokens
def get_token_sparse(datas, Nbatch,T,voc):
    tokens = np.zeros((Nbatch)).astype('int32')
    for i in range(Nbatch):
        tokens[i,:] = datas[i]
    return tokens

def get_mask_seqinsert_efficient(insertpos, Nbatch,T):
    masks = np.zeros((Nbatch,T,1)).astype('float32')
    for i in range(Nbatch): 
        masks[i, insertpos[i],0] = 1
    return masks

def get_test_batch_efficient2(imgpath2,tokendata2,metaefficient2,h1,w1,T,cnn,hw,dimimg,Nstep,Nleap,capa,i):
            Nbatch = 0
            imgssmall = np.zeros((Nleap,h1,w1,3), dtype='float32')
            leaplist = np.zeros((Nleap),dtype='int8')
            for k in range(Nleap):
                leaplist[k] = metaefficient2[i+Nbatch]
                img = get_img(imgpath2[i+Nbatch:i+Nbatch+1], 1,h1,w1,3,True)
                imgssmall[k:k+1,:] = np.copy(img)
                Nbatch = Nbatch + metaefficient2[i+Nbatch]
            featuressmall = cnn.predict(imgssmall)
            imgs = np.zeros((np.sum(leaplist),hw,hw,dimimg), dtype='float32')
            pos = 0
            for k in range(Nleap):
                imgs[pos:pos+leaplist[k],:] = np.tile(featuressmall[k:k+1,:], [leaplist[k],1,1,1])
                pos = pos+leaplist[k]
            token1 = np.asarray(tokendata2[i:i+Nbatch], dtype='int32')
            token2 = token1[:, 1:]
            token1 = token1[:,0:-1]
            masks = (token2>0).astype('float32')
            posembed = np.expand_dims(np.eye(T), axis=0)
            posembeds = np.tile(posembed, [Nbatch,1,1])
            kbpseudo = np.zeros((Nbatch,capa,1), dtype='float32')
            querypseudo = np.ones((Nbatch,capa), dtype='int32')
            querypseudo = np.concatenate((querypseudo, token1), axis=-1)
            return imgs,token1, kbpseudo, posembeds, querypseudo, token2, masks, imgssmall

def get_train_batch_efficient2(imgpath2,tokendata2,metaefficient2,h1,w1,T,cnn,hw,dimimg,Nstep,Nleap,capa):
    while 1:
        i = 0 
        for j in range(0, Nstep):
            Nbatch = 0
            imgssmall = np.zeros((Nleap,h1,w1,3), dtype='float32')
            leaplist = np.zeros((Nleap),dtype='int8')
            for k in range(Nleap):
                leaplist[k] = metaefficient2[i+Nbatch]
                img = get_img(imgpath2[i+Nbatch:i+Nbatch+1], 1,h1,w1,3,True)
                imgssmall[k:k+1,:] = np.copy(img)
                Nbatch = Nbatch + metaefficient2[i+Nbatch]
            featuressmall = cnn.predict(imgssmall)
            imgs = np.zeros((np.sum(leaplist),hw,hw,dimimg), dtype='float32')
            pos = 0
            for k in range(Nleap):
                imgs[pos:pos+leaplist[k],:] = np.tile(featuressmall[k:k+1,:], [leaplist[k],1,1,1])
                pos = pos+leaplist[k]
            token1 = np.asarray(tokendata2[i:i+Nbatch], dtype='int32')
            token2 = token1[:, 1:]
            token1 = token1[:,0:-1]
            masks = (token2>0).astype('float32')
            posembed = np.expand_dims(np.eye(T), axis=0)
            posembeds = np.tile(posembed, [Nbatch,1,1])
            kbpseudo = np.zeros((Nbatch,capa,1), dtype='float32')
            querypseudo = np.ones((Nbatch,capa), dtype='int32')
            querypseudo = np.concatenate((querypseudo, token1), axis=-1)
            
            i = i+Nbatch
            yield([imgs,token1, kbpseudo, posembeds, querypseudo], {'lnow': token2}, masks)













def get_train_insertion(imgpath2,insdata2,inskey2,inspos2,insmeta2,h1,w1,T,Tk,cnn,hw,dimimg,Nstep,Nleap,capa):
    while 1:
        i = 0 
        for j in range(0, Nstep):
            Nbatch = 0
            imgssmall = np.zeros((Nleap,h1,w1,3), dtype='float32')
            leaplist = np.zeros((Nleap),dtype='int8')
            for k in range(Nleap):
                leaplist[k] = insmeta2[i+Nbatch]
                img = get_img(imgpath2[i+Nbatch:i+Nbatch+1], 1,h1,w1,3,True)
                imgssmall[k:k+1,:] = np.copy(img)
                Nbatch = Nbatch + insmeta2[i+Nbatch]
            featuressmall = cnn.predict(imgssmall)
            imgs = np.zeros((np.sum(leaplist),hw,hw,dimimg), dtype='float32')
            pos = 0
            for k in range(Nleap):
                imgs[pos:pos+leaplist[k],:] = np.tile(featuressmall[k:k+1,:], [leaplist[k],1,1,1])
                pos = pos+leaplist[k]
            token1 = np.asarray(insdata2[i:i+Nbatch], dtype='int32')
            token2 = token1[:, 1:]
            token1 = token1[:,0:-1]
            key1 = np.asarray(inskey2[i:i+Nbatch], dtype='int32')
            
            masks = (token2>0).astype('float32')
            pos2 = inspos2[i:i+Nbatch]
            for k in range(Nbatch):
                masks[k,0:max(0,pos2[k]-1)] = 0
            
            posembed = np.expand_dims(np.eye(T), axis=0)
            posembeds = np.tile(posembed, [Nbatch,1,1])
            posembed2 = np.expand_dims(np.eye(Tk), axis=0)
            posembed2s = np.tile(posembed2, [Nbatch,1,1])
            kbpseudo = np.zeros((Nbatch,capa,1), dtype='float32')
            querypseudo = np.ones((Nbatch,capa), dtype='int32')
            querypseudo = np.concatenate((querypseudo, token1), axis=-1)
            
            i = i+Nbatch
            yield([imgs,token1,kbpseudo,posembeds,querypseudo,key1,posembed2s], {'lnow': token2}, masks)

def get_test_insertion(imgpath2,insdata2,inskey2,inspos2,insmeta2,h1,w1,T,Tk,cnn,hw,dimimg,Nstep,Nleap,capa,i):
        for j in range(0, Nstep):
            Nbatch = 0
            imgssmall = np.zeros((Nleap,h1,w1,3), dtype='float32')
            leaplist = np.zeros((Nleap),dtype='int8')
            for k in range(Nleap):
                leaplist[k] = insmeta2[i+Nbatch]
                img = get_img(imgpath2[i+Nbatch:i+Nbatch+1], 1,h1,w1,3,True)
                imgssmall[k:k+1,:] = np.copy(img)
                Nbatch = Nbatch + insmeta2[i+Nbatch]
            featuressmall = cnn.predict(imgssmall)
            imgs = np.zeros((np.sum(leaplist),hw,hw,dimimg), dtype='float32')
            pos = 0
            for k in range(Nleap):
                imgs[pos:pos+leaplist[k],:] = np.tile(featuressmall[k:k+1,:], [leaplist[k],1,1,1])
                pos = pos+leaplist[k]
            token1 = np.asarray(insdata2[i:i+Nbatch], dtype='int32')
            token2 = token1[:, 1:]
            token1 = token1[:,0:-1]
            key1 = np.asarray(inskey2[i:i+Nbatch], dtype='int32')
            
            masks = (token2>0).astype('float32')
            pos2 = inspos2[i:i+Nbatch]
            for k in range(Nbatch):
                masks[k,0:max(0,pos2[k]-1)] = 0
            
            posembed = np.expand_dims(np.eye(T), axis=0)
            posembeds = np.tile(posembed, [Nbatch,1,1])
            posembed2 = np.expand_dims(np.eye(Tk), axis=0)
            posembed2s = np.tile(posembed2, [Nbatch,1,1])
            kbpseudo = np.zeros((Nbatch,capa,1), dtype='float32')
            querypseudo = np.ones((Nbatch,capa), dtype='int32')
            querypseudo = np.concatenate((querypseudo, token1), axis=-1)
            
            return imgs,token1,kbpseudo,posembeds,querypseudo,key1,posembed2s, token2, masks, imgssmall


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
