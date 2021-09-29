#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.interpolate import *




# In[ ]:


def integral(x, info , constant=0):
    x = np.atleast_1d(x)           #convertir x en 1d list 
    
    out = np.zeros(x.shape, dtype=x.dtype)
    for n in range(len(out)):
        out[n] = splint(0, x[n], info)       #Integral 
    out += constant
    return out


# In[ ]:


# c'est la phase de la normalisation des courbes paramétrés par longueur d'arc normalisée 


def Reparametrage_euclidien(x,y,n):
    
    nx=(len(x)-1)
    t = list(np.arange(0,1,1/nx))
    if(len(t) != len(x)):
        t.append(1.)
    
    
   
    x_t = splrep(t,x, s=0) # x_t_info:un tuple contenant les nœuds, les coefficients et le degré de la fonction 
    y_t = splrep(t,y, s=0)
    
       
    l,s = AbscisseEuc(t,x_t,y_t,n)
    
    X1 = splev(s,x_t, der=0)
    Y1 = splev(s,y_t, der=0)
    
    

    X1 = X1/l           # eliminer le facteur d'echelle 
    Y1 = Y1/l

    X1 = X1-(sum(X1)/len(X1))         # eliminer la translation 
    Y1 = Y1 -(sum(Y1)/len(Y1))
    
    return  X1 , Y1
    
    


# In[ ]:


def AbscisseEuc(t1,p1,p2,N) :
    
    x_t_derive = splev(t1,p1, der=1) # derivé de la fonction x(t) au points t1 (liste de point)
    y_t_derive = splev(t1,p2, der=1) # derivé de la fonction y(t)
    
    
    x_t_deriver2 = [number ** 2 for number in x_t_derive] # le carre des valeurs
    y_t_deriver2 = [number ** 2 for number in y_t_derive]
    
    s_derive =[math.sqrt(x + y) for x, y in zip(x_t_deriver2, y_t_deriver2 )]
  
    
    s_derive_info = splrep(t1,s_derive, s=0) # s'(t)
    s_t = integral(t1,s_derive_info)     # s(t)
    
    L = max ([abs(number) for number in s_t])
    s_t = s_t/L                #normalisation
    
    
    s_t , indices = np.unique(s_t , return_index=True)
    nv_t = [ t1[i] for i in indices ]
    
    s = interp1d(s_t,nv_t)
    a=N-1
    pas = list(np.arange(0,1,1./a))
    if(len(pas) != N) :
        pas.append(1.)
    
    return L,s(pas)

