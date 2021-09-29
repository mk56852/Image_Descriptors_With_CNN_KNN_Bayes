#!/usr/bin/env python
# coding: utf-8

# # L'importation des bibliothéques 

# In[13]:


import data as d
import classifieurs as cl
import suivi as s 
import reparametrage_euc as rep
import cv2
import descripteurs as des
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model


# # Importation d'image 

# In[14]:


img = cv2.imread("C:/Users/dell/Desktop/PCD/PNG_data2/bat-3.png")
plt.imshow(img)


# # Filtre de Canny 

# In[15]:


img_canny =  cv2.Canny(img,50,100)
plt.imshow(img_canny)


# # La dilatation d'image

# In[16]:


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
img_cnt = cv2.dilate(img_canny, kernel)
plt.imshow(img_cnt)


# # Suivi de Contour

# In[17]:


contour = s.pavlidis(img_cnt)


# In[18]:


x , y = s.getXY(contour)


# In[19]:


get_ipython().run_line_magic('matplotlib', 'notebook')

from matplotlib import animation


fig = plt.figure()
ax = plt.axes(xlim=(0, 700), ylim=(0, 800))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):

    line.set_data(y[:20*i], x[:20*i])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=200, blit=True)


# # Le Reparamétrage

# In[20]:


X1,Y1 = rep.Reparametrage_euclidien(x,y,100)

plt.scatter (Y1,X1)


# # Calcul de descripteur 

# In[21]:


descripteur1  = np.array(des.norme(X1,Y1))
descripteur = np.absolute(descripteur1)
descripteur = descripteur.reshape(1,descripteur.shape[0],1)


# # La prédiction 

# ###            1 ) l'importation du model

# In[22]:



CNNmodel = load_model('CNNmodel.h5')


# ### 2) l'importation du Labels 

# In[23]:


datalabel = d.getDataDict()


# ### 3) la prediction 

# In[24]:


pred = CNNmodel.predict(descripteur)
id_classe = list(pred[0]).index(max(pred[0]))
print("CNNmodel prediction : -> " + datalabel[id_classe])


# In[ ]:




