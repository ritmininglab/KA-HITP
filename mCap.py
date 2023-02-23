import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import ReLU, Concatenate, Conv1D, RepeatVector, Embedding, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Reshape,Softmax
from tensorflow.keras.layers import MultiHeadAttention
from keras_layer_normalization import LayerNormalization

from tensorflow.keras.models import load_model, Model

from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from tensorflow.keras.layers import LSTM as RNN

from tensorflow.keras.utils import to_categorical

initialvar = -8
small = 1e-5

class Slice(layers.Layer):
    def __init__(self, nbbox, name='name'):
        super(Slice, self).__init__(name=name)
        self.nbbox = nbbox
    def call(self, x):
        xs = tf.split(x, num_or_size_splits=self.nbbox, axis=2)
        return xs
class Concat(layers.Layer):
    def __init__(self, nbbox, name='name'):
        super(Concat, self).__init__(name=name)
        self.nbbox = nbbox
    def call(self, x):
        x2 = []
        for i in range(self.nbbox):
            x2.append(tf.expand_dims(x[i], axis=1))
        xs = tf.concat(x2, axis=1)
        return xs
    def compute_output_shape(self, inshape):
        return (inshape[0][0], self.nbbox, inshape[0][1])


class Encode(layers.Layer):
    def __init__(self, thigh,tlow,divide, name='name'):
        super().__init__(name=name)
        self.thigh = thigh
        self.tlow = tlow
        self.divide = divide
    def call(self, x):
        xa = x[0]+x[1]
        term1 =  tf.nn.relu(self.thigh-xa) * x[2]
        term2 =  tf.nn.relu(xa-self.tlow) * (1-x[2])
        return (term1+term2)/self.divide
    def compute_output_shape(self, inshape):
        return inshape[0]



class Conv(layers.Layer):
    def __init__(self, dims, trainable=True, idx=0, wreg=1e-5,name='name'):
        super().__init__(name=name)
        self.dim1 = dims[idx]
        self.dim2 = dims[idx+1]
        self.trainable = trainable
        self.wreg = wreg
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.025)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[3,3,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
    def call(self, x):
        term0 = tf.add(tf.nn.conv2d(x, self.w, [1,1,1,1], padding='SAME'), self.b)
        
        l2loss = self.wreg* (tf.reduce_sum(tf.square(self.w)) +tf.reduce_sum(tf.square(self.b)))
        self.add_loss(l2loss)
        return term0
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)


class Masks(layers.Layer):
    def __init__(self, name='name'):
        super().__init__(name=name)
    def call(self, x):
        tokenembed = x[0]
        tokenseq = x[1]
        mask0 = tf.math.not_equal(tokenseq, 0)
        padding_mask = tf.cast(mask0[:, :, tf.newaxis], dtype=tf.int32)
        combined_mask = tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32)
        
        input_shape = tf.shape(tokenembed)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        causal_mask = tf.tile(mask, mult)
        combined_mask = tf.minimum(combined_mask, causal_mask)
        """
        positions = tf.range(start=0, limit=self.T, delta=1)
        positions = to_categorical(positions, self.T)
        self.add_loss(l2loss)
        """
        return [padding_mask, combined_mask]
    def compute_output_shape(self, inshape):
        return [(inshape[0], inshape[1],1), (inshape[0], inshape[1],inshape[1])]

class AttnCast1(layers.Layer):
    def __init__(self, T,hw, dimimg,dimtxt,name='name'):
        super(AttnCast1, self).__init__(name=name)
        self.T = T
        self.hw = hw
        self.dimimg = dimimg
        self.dimtxt = dimtxt
    def build(self, inshape):
        self.nbatch = inshape[0]
    def call(self, x):
        fmap = x 
        fmap = tf.expand_dims(fmap,axis=1)
        fmap = tf.tile(fmap, [1,self.T,1,1,1])
        return fmap
    def compute_output_shape(self, inshape):
        return (self.nbatch, self.T,self.hw,self.hw, self.dimimg)
class AttnCast2(layers.Layer):
    def __init__(self, T,hw, dimimg,dimtxt,name='name'):
        super(AttnCast2, self).__init__(name=name)
        self.T = T
        self.hw = hw
        self.dimimg = dimimg
        self.dimtxt = dimtxt
    def build(self, inshape):
        self.nbatch = inshape[0][0]
    def call(self, x):
        fmap = x[0] 
        ftxt = x[1] 
        ftxt = tf.expand_dims(ftxt,axis=2)
        ftxt = tf.expand_dims(ftxt,axis=3)
        ftxt = tf.tile(ftxt, [1,1,self.hw,self.hw,1])
        fcat = tf.concat([fmap,ftxt], axis=-1)
        return fcat
    def compute_output_shape(self, inshape):
        return (self.nbatch, self.T,self.hw,self.hw, self.dimimg+self.dimtxt)

class AttnSoftmax(layers.Layer):
    def __init__(self, T,hw,nbatch, name='name'):
        super(AttnSoftmax, self).__init__(name=name)
        self.T = T
        self.hw = hw
        self.nbatch = nbatch
    def call(self, x):
        """
        temp = tf.reshape(x[:,:,:,:,0], [self.nbatch, self.T, self.hw*self.hw])
        temp = tf.nn.softmax(temp)
        temp = tf.reshape(temp, [self.nbatch, self.T, self.hw, self.hw])
        return tf.expand_dims(temp, axis=-1)
        """
        temp = tf.reshape(x, [self.nbatch, self.T, self.hw*self.hw, 1])
        temp = tf.nn.softmax(temp, axis=-2)
        temp = tf.reshape(temp, [self.nbatch, self.T, self.hw, self.hw, 1])
        return temp
    def compute_output_shape(self, inshape):
        return inshape

class AttnPool(layers.Layer):
    def __init__(self, name='name'):
        super(AttnPool, self).__init__(name=name)
    def call(self, x):
        fmap = x[0]
        masks = x[1]
        temp = fmap*masks
        output = tf.reduce_sum(temp, axis=[2,3])
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[-1])




class MasksInsertionEfficient(layers.Layer):
    def __init__(self, T, name='name'):
        super().__init__(name=name)
        self.T = T
    def call(self, x):
        tokenseq = x
        mask0 = tf.math.not_equal(tokenseq, 0)
        padding_mask = tf.cast(mask0[:, :, tf.newaxis], dtype=tf.int32)
        combined_mask = tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32)
        
        temp1 = tf.tile(tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32), [1,self.T,1])
        temp2 = tf.tile(tf.cast(padding_mask, dtype=tf.int32), [1,1,self.T])
        combined_mask = tf.minimum(temp1,temp2)
        
        return combined_mask
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1],inshape[1])


class Position(layers.Layer):
    def __init__(self, nbatch,T, name='name'):
        super().__init__(name=name)
        self.nbatch = nbatch
        self.T = T
        self.position = tf.range(start=0, limit=T, delta=1)
        self.positions = tf.tile(tf.expand_dims(self.position, 0), [self.nbatch, 1])
    def call(self, x):
        return self.positions
    def compute_output_shape(self, inshape):
        return (self.nbatch,self.T)

class Shift1Step(layers.Layer):
    def __init__(self, name='name'):
        super().__init__(name=name)
    def build(self, inshape):
        self.nbatch = inshape[0]
        self.T = inshape[1]
        self.dim = inshape[2]
    def call(self, x):
        laststep = tf.zeros((self.nbatch,1,self.dim))
        x1toT = x[:,1:,:]
        output = tf.concat([x1toT, laststep], axis=-2)
        return output
    def compute_output_shape(self, inshape):
        return inshape





class PositionCat(layers.Layer):
    def __init__(self, nbatch,T, name='name'):
        super().__init__(name=name)
        self.nbatch = nbatch
        self.T = T
        self.position = tf.range(start=0, limit=T, delta=1)
        self.position = tf.one_hot(self.position, T)
        self.positions = tf.tile(tf.expand_dims(self.position, 0), [self.nbatch, 1,1])
    def call(self, x):
        return self.positions
    def compute_output_shape(self, inshape):
        return (self.nbatch,self.T,self.T)
    
class TokenCat(layers.Layer):
    def __init__(self, nbatch,T,vocsize, name='name'):
        super().__init__(name=name)
        self.nbatch = nbatch
        self.T = T
        self.vocsize = vocsize
    def call(self, x):
        return tf.one_hot(x, self.vocsize)
    def compute_output_shape(self, inshape):
        return (self.nbatch,self.T,self.vocsize)



class Embed2(layers.Layer):
    def __init__(self, dim2, priors, trn, idx, kldiv, name='name'):
        super().__init__(name=name)
        self.dim2 = dim2
        self.trainable = trn
        self.wmu0 = priors[idx][0]
        self.wvar0 = priors[idx][1]
        self.bmu0 = priors[idx][2]
        self.bvar0 = priors[idx][3]
        self.kldiv = kldiv
    def build(self, inshape):
        self.dim1 = inshape[-1]
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        
        output = tf.add(tf.matmul(x, wnow), bnow)
        term0 = -0.5*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], self.dim2)




class MaskCausal(layers.Layer):
    def __init__(self, nbatch,T,name='name'):
        super().__init__(name=name)
        self.nbatch = nbatch
        self.T = T
    def call(self, x):
        tokenseq = x
        mask0 = tf.math.not_equal(tokenseq, 0)
        combined_mask = tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32)
        
        i = tf.range(self.T)[:, tf.newaxis]
        j = tf.range(self.T)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, self.T, self.T))
        causal_mask = tf.tile(mask, [self.nbatch,1,1])
        combined_mask = tf.minimum(combined_mask, causal_mask)
        return combined_mask
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1],inshape[1])





def AtnTransformerOld(datas, dims2, T, trainables, cnn, hw,dimimg,Nbatch, bnparam):
    embed_dim = dims2[0]
    ff_dim = dims2[1]
    vocsize = dims2[-1]
    priors = bnparam[0]
    kldiv = bnparam[1]
    trn1 = bnparam[2][0]
    trn2 = bnparam[2][1]

    
    vgg = cnn(datas[0])
    f5 = Reshape((-1, dimimg), name='f5')(vgg)
    f5a = Embed(embed_dim, priors, trn2, 2, kldiv, name='f5a')(f5)
    f5b = LayerNormalization(trainable=trn2,name='f5b')(f5a)
    f5c = ReLU(name='f5c')(f5b)
    avgimg = Lambda(lambda x: tf.reduce_mean(x, axis=-2, keepdims=True), name='avgimg')(f5)
    f5aavg = Embed(embed_dim, priors, trn2, 3, kldiv, name='f5aavg')(avgimg)
    f5bavg = LayerNormalization(trainable=trn2, name='f5bavg')(f5aavg)
    f5cavg = ReLU(name='f5cavg')(f5bavg)
    
    tokens = TokenCat(Nbatch,T,vocsize, name='tokens')(datas[1])
    embed = Embed(embed_dim, priors, trn1, 0, kldiv, name='embed')(tokens)
    positions = PositionCat(Nbatch,T,name='positions')(datas[1])
    embed2 = Embed(embed_dim, priors, trn1, 1, kldiv, name='embed2')(positions)
    embedpos = Lambda(lambda x: x[0]+x[1], name='embedpos') ([embed, embed2])
    
    mask = MaskCausal(Nbatch,T,name='mask')(datas[1])
    attn1 = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn1')(query=embedpos, value=embedpos, key=embedpos, attention_mask=mask)
    out10 = Lambda(lambda x: x[0]+x[1], name='out10') ([embed, attn1])
    out1 =LayerNormalization(trainable=trn2,name='out1') (out10)
    """
    attn2 = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn2')(query=out1, value=out1, key=out1, attention_mask=mask)
    out20 = Lambda(lambda x: x[0]+x[1], name='out20') ([out1, attn2])
    out2 =LayerNormalization(trainable=trn2,name='out2') (out20)
    """
    lrnn01toT = Lambda(lambda x: x[:,1:,:], name='lrnn01toT')(out1)
    f5a1 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=1), [1,T-1,1,1]), name='f5a1')(f5c)
    f5a2 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=2), [1,1,hw*hw,1]), name='f5a2')(lrnn01toT)
    f5a3 = Concatenate(axis=-1, name='f5a3')([f5a1,f5a2])
    f6b = Embed(ff_dim, priors, trn2, 4, kldiv, name='f6b')(f5a3)
    f6b0 = LayerNormalization(trainable=trn2, name='f6b0')(f6b)
    f6b0r = ReLU(name='f6b0r')(f6b0)
    f6b1 = Embed(1, priors, trn2, 5, kldiv, name='f6b1')(f6b0r)
    f6b2 = Softmax(axis=-2, name='f6b2')(f6b1)
    f6b3 = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-2), name='f6b3')([f5a1, f6b2])
    cattime = Concatenate(axis=-2, name='cattime')([f5cavg, f6b3])
    
    imgtxtembed = Concatenate(axis=-1, name='imgtxtembed')([out1, cattime])
    dense0 = Embed(embed_dim, priors, trn2, 6, kldiv, name='dense0')(imgtxtembed)
    dense01 = LayerNormalization(trainable=trn2, name='dense01') (dense0)
    dense02 = ReLU(name='dense02')(dense01)
    
    lnow = Embed(vocsize, priors, trn2, 7, kldiv, name='lnow')(dense02)
    
    return [lnow, f6b2]


class MasksInsertion(layers.Layer):
    def __init__(self, T, name='name'):
        super().__init__(name=name)
        self.T = T
    def call(self, x):
        tokenseq = x
        mask0 = tf.math.not_equal(tokenseq, 0)
        padding_mask = tf.cast(mask0[:, :, tf.newaxis], dtype=tf.int32)
        combined_mask = tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32)
        
        temp1 = tf.tile(tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32), [1,self.T,1])
        temp2 = tf.tile(tf.cast(padding_mask, dtype=tf.int32), [1,1,self.T])
        combined_mask = tf.minimum(temp1,temp2)
        
        return [padding_mask, combined_mask]
    def compute_output_shape(self, inshape):
        return [(inshape[0], inshape[1],1), (inshape[0], inshape[1],inshape[1])]

"""
class Lcrop(layers.Layer):
    def __init__(self, name='name'):
        super().__init__(name=name)
    def build(self, inshape):
        self.nbatch = inshape[0][0]
        self.T = inshape[0][1]
        self.dim = inshape[0][2]
    def call(self, x):
        temp = tf.multiply(x[0],x[1])
        output = tf.reduce_sum(temp,axis=-2, keepdims=True)
        return output
    def compute_output_shape(self, inshape):
        return (self.nbatch,1,self.dim)

class Shift1StepEfficient(layers.Layer):
    def __init__(self, name='name'):
        super().__init__(name=name)
    def build(self, inshape):
        self.nbatch = inshape[0]
        self.T = inshape[1]
        self.dim = inshape[2]
    def call(self, x):
        x1toT = x[:,1:,:]
        x0 = x[:,0:1,:]
        output = tf.concat([x1toT, x0], axis=-2)
        return output
    def compute_output_shape(self, inshape):
        return inshape
"""

class TokenCatEfficient(layers.Layer):
    def __init__(self, T,vocsize, name='name'):
        super().__init__(name=name)
        self.T = T
        self.vocsize = vocsize
    def call(self, x):
        return tf.one_hot(x, self.vocsize)
    def compute_output_shape(self, inshape):
        return (inshape[0],self.T,self.vocsize)

class Embed(layers.Layer):
    def __init__(self, dim2, priors, trn, idx, kldiv, name='name'):
        super().__init__(name=name)
        self.dim2 = dim2
        self.trainable = trn
        self.wmu0 = priors[idx][0]
        self.wvar0 = priors[idx][1]
        self.bmu0 = priors[idx][2]
        self.bvar0 = priors[idx][3]
        self.kldiv = kldiv
    def build(self, inshape):
        self.dim1 = inshape[-1]
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[1,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[1,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([1,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        
        output = tf.add(tf.nn.conv1d(x, wnow, 1, padding='SAME'), bnow)
        term0 = -0.5*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], self.dim2)
class EmbedKB(layers.Layer):
    def __init__(self, capa,dim2, priors, trn, idx, kldiv, name='name'):
        super().__init__(name=name)
        self.dim1 = capa 
        self.dim2 = dim2 
        self.trainable = trn
        self.wmu0 = priors[idx][0]
        self.wvar0 = priors[idx][1]
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[1,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[1,self.dim1, self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([1,self.dim1,self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        
        output = tf.add(x, wnow)
        term0 = -0.5*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        sumkl = term1 + term2 + term0
        self.add_loss(sumkl / self.kldiv)
        
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], self.dim2)
    

class zlayershare(layers.Layer):
    def __init__(self, dim2, priors, trainable, idx,kldiv,name='layername'):
        super(zlayershare, self).__init__(name=name)
        self.dim2 = dim2
        self.priora = priors[idx][0]
        self.priorb = priors[idx][1]
        self.kldiv = kldiv
        self.trainable = trainable
        self.temperature = 1
    def build(self, inshape): 
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(0.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.dim2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.dim2],)
        self.pienk = self.add_weight("pienk",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[1,self.dim2],)
    def call(self, x):
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.1, maxval=0.9)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        priorlogpie = tf.math.log(kuma)
        
        pienk = tf.sigmoid(self.pienk)
        epspie = tf.random.uniform(shape=tf.shape(pienk), minval=0.1, maxval=0.9)
        temp2 = tf.math.log(epspie)-tf.math.log(1-epspie)
        logpienk = tf.math.log(pienk+small) - tf.math.log(1-pienk+small)
        softbernoulli = 1/(1 + tf.exp(-(logpienk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.priora)+tf.math.lgamma(self.priorb)-tf.math.lgamma(self.priora+self.priorb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.priora, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.priorb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.priora+self.priorb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kldiv) )
        
        temp = pienk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogpie) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogpie) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kldiv) )
        
        result = tf.multiply(x, softbernoulli)
        return result
    def compute_output_shape(self, inshape):
        return inshape
class zlayershareKB(layers.Layer):
    def __init__(self, dim2, priors, trainable, idx,kldiv,name='layername'):
        super(zlayershareKB, self).__init__(name=name)
        self.dim2 = dim2
        self.priora = priors[idx][0]
        self.priorb = priors[idx][1]
        self.kldiv = kldiv
        self.trainable = trainable
        self.temperature = 1
    def build(self, inshape): 
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(0.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.dim2,1],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.dim2,1],)
        self.pienk = self.add_weight("pienk",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[1,self.dim2,1],)
    def call(self, x):
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.1, maxval=0.9)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        priorlogpie = tf.math.log(kuma)
        
        pienk = tf.sigmoid(self.pienk)
        epspie = tf.random.uniform(shape=tf.shape(pienk), minval=0.1, maxval=0.9)
        temp2 = tf.math.log(epspie)-tf.math.log(1-epspie)
        logpienk = tf.math.log(pienk+small) - tf.math.log(1-pienk+small)
        softbernoulli = 1/(1 + tf.exp(-(logpienk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.priora)+tf.math.lgamma(self.priorb)-tf.math.lgamma(self.priora+self.priorb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.priora, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.priorb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.priora+self.priorb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kldiv) )
        
        temp = pienk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogpie) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogpie) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kldiv) )
        
        result = tf.multiply(x, softbernoulli)
        return result
    def compute_output_shape(self, inshape):
        return inshape


class MaskCross(layers.Layer):
    def __init__(self, Tcap,Tkey, name='name'):
        super().__init__(name=name)
        self.Tcap = Tcap
        self.Tkey = Tkey
    def call(self, x):
        maskcap = tf.math.not_equal(x[0], 0)
        maskkey = tf.math.not_equal(x[1], 0)
        
        temp1 = tf.tile(tf.cast(maskcap[:, :,tf.newaxis], dtype=tf.int32), [1,1,self.Tkey])
        temp2 = tf.tile(tf.cast(maskkey[:, tf.newaxis,:], dtype=tf.int32), [1,self.Tcap,1])
        combined_mask = tf.minimum(temp1,temp2)
        
        return combined_mask
    def compute_output_shape(self, inshape):
        return (inshape[0][0], inshape[0][1], inshape[1][1])
class MasksKBuseless(layers.Layer):
    def __init__(self, T,capacity, name='name'):
        super().__init__(name=name)
        self.T = T
        self.cap = capacity
    def call(self, x):
        tokenseq = x[0]
        queryseq = x[1]
        mask0 = tf.math.not_equal(tokenseq, 0)
        maskq = tf.math.not_equal(queryseq, 0)
        padding_mask = tf.cast(maskq[:, :, tf.newaxis], dtype=tf.int32)
        
        temp1 = tf.tile(tf.cast(mask0[:, tf.newaxis, :], dtype=tf.int32), [1,self.T+self.cap,1])
        temp2 = tf.tile(tf.cast(padding_mask, dtype=tf.int32), [1,1,self.T])
        combined_mask = tf.minimum(temp1,temp2)
        
        return combined_mask
    def compute_output_shape(self, inshape):
        return (inshape[0][0], inshape[1][1], inshape[0][1])

class MaskQ(layers.Layer):
    def __init__(self, Tconcat, name='name'):
        super().__init__(name=name)
        self.T = Tconcat
        i = tf.range(self.T)[:, tf.newaxis]
        j = tf.range(self.T)
        mask = tf.cast(i >= j, dtype="int32")
        self.trimask = tf.reshape(mask, (1, self.T, self.T))
        
    def call(self, x):
        mask0 = tf.math.not_equal(x, 0)
        
        temp2 = tf.tile(tf.cast(mask0[:, :, tf.newaxis], dtype=tf.int32), [1,1,self.T])
        temp1 = 0*temp2 + self.trimask
        combined_mask = tf.minimum(temp1,temp2)
        
        return combined_mask
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[1])




def AtnKeywordTransformer(datas, dims, T, hw,dimimg, bnparam):
    embed_dim = dims[0]
    ff_dim = dims[1]
    vocsize = dims[2]
    capa = dims[3]
    priors = bnparam[0][0]
    priors2 = bnparam[0][1]
    kldiv = bnparam[1]
    trn1 = bnparam[2][0]
    trn2 = bnparam[2][1]

    f5c = Reshape((-1, dimimg), name='f5c')(datas[0])
    f5cavg = Lambda(lambda x: tf.reduce_mean(x, axis=-2, keepdims=True), name='f5cavg')(f5c)
    
    tokens = TokenCatEfficient(T,vocsize, name='tokens')(datas[1])
    embed = Embed(embed_dim, priors, trn1, 0, kldiv, name='embed')(tokens)
    embed2 = Embed(embed_dim, priors, trn1, 1, kldiv, name='embed2')(datas[3])
    embedpos = Lambda(lambda x: x[0]+x[1], name='embedpos') ([embed, embed2])
    embed3a = EmbedKB(capa, embed_dim, priors, trn1, 2, kldiv, name='embed3a')(datas[2])
    embed3 = zlayershareKB(capa, priors2, trn1, 0,kldiv,name='embed3')(embed3a)
    embedq = Lambda(lambda x: tf.concat([x[0],x[1]], axis=-2), name='embedq') ([embed3,embedpos])

    mask = MaskQ(T+capa,name='mask')(datas[4])
    attn1a = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn1a')(query=embedq, value=embedq, key=embedq, attention_mask=mask)
    attn1 = Lambda(lambda x: x[:,capa:], name='attn1') (attn1a)
    out10 = Lambda(lambda x: x[0]+x[1], name='out10') ([embedpos, attn1])
    out1 = LayerNormalization(trainable=trn2,name='out1') (out10)
    fd1 = Embed(embed_dim, priors, trn2, 3, kldiv, name='fd1')(out1)
    fd1a = ReLU(name='fd1a')(fd1)
    fd1b = zlayershare(embed_dim, priors2, trn1, 1,kldiv,name='fd1b')(fd1a)
    fd1c = Lambda(lambda x: x[0]+x[1], name='fd1c') ([out1,fd1b])
    fd1d = LayerNormalization(trainable=trn2,name='fd1d') (fd1c)
    
    lrnn01toT = Lambda(lambda x: x[:,1:,:], name='lrnn01toT')(fd1d)
    f5a1 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=1), [1,T-1,1,1]), name='f5a1')(f5c)
    f5a2 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=2), [1,1,hw*hw,1]), name='f5a2')(lrnn01toT)
    f5a3 = Concatenate(axis=-1, name='f5a3')([f5a1,f5a2])
    f5b = Embed(ff_dim, priors, trn2, 4, kldiv, name='f5b')(f5a3)
    f5b0 = LayerNormalization(name='f5b0')(f5b)
    f5b1 = Embed(1, priors, trn2, 5, kldiv, name='f5b1')(f5b0)
    f5b2 = Softmax(axis=-2, name='f5b2')(f5b1)
    f5d = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=2), name='f5d')([f5a1, f5b2])
    
    cattime = Concatenate(axis=-2, name='cattime')([f5cavg, f5d])
    
    imgtxtembed = Concatenate(axis=-1, name='imgtxtembed')([fd1d,cattime])
    dense0 = Embed(embed_dim, priors, trn2, 6, kldiv, name='dense0')(imgtxtembed)
    dense01 = LayerNormalization(trainable=trn2, name='dense01') (dense0)
    dense02 = ReLU(name='dense02')(dense01)
    
    lnow = Embed(vocsize, priors, trn2, 7, kldiv, name='lnow')(dense02)
    return [lnow, f5d]



def AtnInsertTransformer(datas, dims, T,Tk, hw,dimimg, bnparam):
    embed_dim = dims[0]
    ff_dim = dims[1]
    vocsize = dims[2]
    capa = dims[3]
    priors = bnparam[0][0]
    priors2 = bnparam[0][1]
    kldiv = bnparam[1]
    trn1 = bnparam[2][0]
    trn2 = bnparam[2][1]

    f5c = Reshape((-1, dimimg), name='f5c')(datas[0])
    f5cavg = Lambda(lambda x: tf.reduce_mean(x, axis=-2, keepdims=True), name='f5cavg')(f5c)
    
    tokens = TokenCatEfficient(T,vocsize, name='tokens')(datas[1])
    embed = Embed(embed_dim, priors, trn1, 0, kldiv, name='embed')(tokens)
    embed2 = Embed(embed_dim, priors, trn1, 1, kldiv, name='embed2')(datas[3])
    embedpos = Lambda(lambda x: x[0]+x[1], name='embedpos') ([embed, embed2])
    embed3a = EmbedKB(capa, embed_dim, priors, trn1, 2, kldiv, name='embed3a')(datas[2])
    embed3 = zlayershareKB(capa, priors2, trn1, 0,kldiv,name='embed3')(embed3a)
    embedq = Lambda(lambda x: tf.concat([x[0],x[1]], axis=-2), name='embedq') ([embed3,embedpos])

    mask = MaskQ(T+capa,name='mask')(datas[4])
    attn1a = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn1a')(query=embedq, value=embedq, key=embedq, attention_mask=mask)
    attn1 = Lambda(lambda x: x[:,capa:], name='attn1') (attn1a)
    out10 = Lambda(lambda x: x[0]+x[1], name='out10') ([embedpos, attn1])
    out1 = LayerNormalization(trainable=trn2,name='out1') (out10)
    fd1 = Embed(embed_dim, priors, trn2, 3, kldiv, name='fd1')(out1)
    fd1a = ReLU(name='fd1a')(fd1)
    fd1b = zlayershare(embed_dim, priors2, trn1, 1,kldiv,name='fd1b')(fd1a)
    fd1c = Lambda(lambda x: x[0]+x[1], name='fd1c') ([out1,fd1b])
    fd1d = LayerNormalization(trainable=trn2,name='fd1d') (fd1c)
    
    
    tokensk = TokenCatEfficient(Tk,vocsize, name='tokensk')(datas[5])
    embedk = Embed(embed_dim, priors, trn1, 4, kldiv, name='embedk')(tokensk)
    embedk2 = Embed(embed_dim, priors, trn1, 5, kldiv, name='embedk2')(datas[6])
    embedposk = Lambda(lambda x: x[0]+x[1], name='embedposk') ([embedk, embedk2])
    
    mask2 = MaskCross(T,Tk,name='mask2')([datas[1],datas[5]])
    attn2a = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn2a')(query=fd1d, value=embedposk, key=embedposk, attention_mask=mask2)
    out20 = Lambda(lambda x: x[0]+x[1], name='out20') ([fd1d, attn2a])
    out2 = LayerNormalization(trainable=trn2,name='out2') (out20)
    fd2 = Embed(embed_dim, priors, trn2, 6, kldiv, name='fd2')(out2)
    fd2a = ReLU(name='fd2a')(fd2)
    fd2b = zlayershare(embed_dim, priors2, trn1, 2,kldiv,name='fd2b')(fd2a)
    fd2c = Lambda(lambda x: x[0]+x[1], name='fd2c') ([out2,fd2b])
    fd2d = LayerNormalization(trainable=trn2,name='fd2d') (fd2c)
    
    
    f4a1 = Lambda(lambda x: tf.tile(x, [1,T,1]), name='f4a1')(f5cavg)
    f4a2 = Concatenate(axis=-1, name='f4a2')([fd2d,f4a1])
    f4b1 = Embed(1, priors, trn2, 7, kldiv, name='f4b1')(f4a2)
    f4b2 = Lambda(lambda x: tf.math.sigmoid(x), name='f4b2')(f4b1)
    
    f6a1 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=1), [1,T,1,1]), name='f6a1')(f5c)
    f6a2 = Lambda(lambda x: tf.tile(tf.expand_dims(x,axis=2), [1,1,hw*hw,1]), name='f6a2')(fd2d)
    f6a3 = Concatenate(axis=-1, name='f6a3')([f6a1,f6a2])
    f6b = Embed(ff_dim, priors, trn2, 8, kldiv, name='f6b')(f6a3)
    f6b0 = LayerNormalization(name='f6b0')(f6b)
    f6b1 = Embed(1, priors, trn2, 9, kldiv, name='f6b1')(f6b0)
    f6b2 = Softmax(axis=-2, name='f6b2')(f6b1)
    f6d = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-2), name='f6d')([f6a1, f6b2])
    f6d1 = Lambda(lambda x: x[0]*x[1], name='f6d1')([f6d, f4b2])
    f6d2 = Lambda(lambda x: x[0]*(1-x[1]), name='f6d2')([fd2d, f4b2])
    
    imgtxtembed = Concatenate(axis=-1, name='imgtxtembed')([f6d1,f6d2])
    dense0 = Embed(embed_dim, priors, trn2, 10, kldiv, name='dense0')(imgtxtembed)
    dense01 = LayerNormalization(trainable=trn2, name='dense01') (dense0)
    dense02 = ReLU(name='dense02')(dense01)
    
    lnow = Embed(vocsize, priors, trn2, 11, kldiv, name='lnow')(dense02)
    return [lnow, f6b2,f4b2, mask]
