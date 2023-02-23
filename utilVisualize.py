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

def getFilter(ra, decay):
    dia = 2*ra+1
    myfilter = np.zeros((dia,dia))
    for r in range(dia):
        for c in range(dia):
            dist2 = (r-ra)**2 + (c-ra)**2
            myfilter[r,c] = np.exp(-dist2 / decay)
    return myfilter

def attn2heatmap(attn, myfilter, enlarge):
    dia = myfilter.shape[0]
    ra = int((dia-1)/2)
    dia2 = attn.shape[0]*enlarge
    output = np.zeros((dia2+2*ra, dia2+2*ra))
    for r in range(attn.shape[0]):
        for c in range(attn.shape[1]):
            h = int(enlarge*(r+0.5))
            w = int(enlarge*(c+0.5))
            h0 = h-ra+ra
            h1 = h+ra+ra
            w0 = w-ra+ra
            w1 = w+ra+ra
            weight = attn[r,c]
            output[h0:h1+1, w0:w1+1] = output[h0:h1+1, w0:w1+1] + weight*myfilter
            
    output = output[ra:dia2+ra, ra:dia2+ra]
    return output


class checkingvalues:
   def __init__(self):
      self.mathfun=[]
     
   def update(self, key):
      found=False
      for i,k in enumerate(self.mathfun):
         if key==k:
            self.mathfun[i]=key
            found=True
            break
      if not found:
         self.mathfun.append(key)
    
   def get(self, key):
      for k in self.mathfun:
         if k==key:
            return True
      return False
   def remove(self, key):
      for i,k in enumerate(self.mathfun):
         if key==k:
            del self.mathfun[i]
 
class HashSet:
   def __init__(self):
      self.key_space = 2096
      self.hash_table=[checkingvalues() for i in range(self.key_space)]
   def hash_values(self, key):
       hash_key=key%self.key_space
       return hash_key
   def add(self, key):
      self.hash_table[self.hash_values(key)].update(key)
   def remove(self, key):
      self.hash_table[self.hash_values(key)].remove(key)
   def contains(self, key):
 
      return self.hash_table[self.hash_values(key)].get(key)
   def display(self):
       ls=[]
       for i in self.hash_table:
           if len(i.mathfun)!=0:ls.append(i.mathfun[0])
       print(ls)
