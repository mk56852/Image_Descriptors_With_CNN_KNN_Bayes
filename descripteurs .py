#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.interpolate import *
from numpy.linalg import eig # pour calculer les valeurs propres 
from sklearn.preprocessing import StandardScaler 
from skimage.feature import hog


# In[4]:


# le descripteur de courbure

def courbure(x,y) :
    
    nx=len(x)-1
    t = list(np.arange(0,1,1/nx))
    if(len(t) != len(x)) :
        t.append(1.)
    
    x_t = splrep(t,x, s=0)   
    y_t = splrep(t,y, s=0)
    
    x_1 = splev(t,x_t, der=1)  # x'(t)
    x_2 = splev(t,x_t, der=2)  # x"(t)
    
    y_1 = splev(t,y_t, der=1)  # y'(t)
    y_2 = splev(t,y_t, der=2)  # y"(t)
    
    
    k = list () 
    for i in range(len(x)) :
        a = (x_1[i] * y_2[i]) - (y_1[i] * x_2[i])
        b = ((x_1[i]**2) + (y_1[i]**2))**1.5
        res = a/b
        k.append(abs(res))
    
    return k


# In[11]:


def norme(xi,yi) :
    
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    fourier = np.fft.fft(zi)
    I =fourier/fourier[1]
    
    return I


# In[11]:


def crimmins(xi,yi,n0,n1):
    
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    tfourier = np.fft.fft(zi)
    fourier = np.fft.fftshift(tfourier)
    I = list()
    for i in range(-len(zi)//2,len(zi)//2) :
        In = (fourier[i]**(n0-n1)) * (fourier[n0-1]**(n1-i)) * (fourier[n1-1]**(i-n0))
        I.append(In)

    return I


# In[9]:


def crimmins_stable(xi,yi,n0,n1,p,q):
    
    zi = [(i * 1j) + h for i,h  in zip(yi,xi)]
    fourier = np.fft.fft(zi)
    I = list()
    for i in range(len(zi)) :
        In = (fourier[i]**(n0-n1)) * (fourier[n0-1]**(n1-i)) * (fourier[n1-1]**(i-n0))
        Jn = (fourier[n0-1]**(n1-i-p)) * (fourier[n1-1]**(i-n0-q))
        I.append(In/Jn)

    return I


# In[8]:


def hog (img) :
    # Il faut redimensionner l'image 
    img = cv2.resize(img, (64,128))
    d = cv2.HOGDescriptor()
    hog = d.compute(img)
    hog=hog.reshape(3780)
    return hog


# In[ ]:




