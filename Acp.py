#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler 
import numpy as np 
from numpy.linalg import eig

def acp(data,nombre_res):
    
    
    
    ss = StandardScaler()                   
    data = ss.fit_transform(data)
    
    matrice_convolution = np.cov(data.T)
    valeurs_propre, vecteurs_propre = eig(matrice_convolution)

    valeurs_propre=(valeurs_propre/valeurs_propre.sum()) # pourcentage d'information sur chaque vecteur propre
    
    index = np.argsort(valeurs_propre)[::-1]
    vecteurs_propre = vecteurs_propre[:,index]
    
   
    
    vecteurs_propre = vecteurs_propre[:, :nombre_res] 
    
    
    #projeter sur la nouvelle base
    resultat = np.dot(vecteurs_propre.T, data.T).T
    
    # calcul du pourcentage d'information sur la nouvelle base 
    pc = sum(valeurs_propre[:nombre_res])
    
    return resultat,pc  


# In[ ]:




