#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


display (os.getcwd())


# In[3]:


os.chdir ('C:Desktop\\Machine Learning\\')
display (os.getcwd())


# In[4]:


df =pd.read_csv("salary.csv")
display (df)


# In[14]:


import math

def d2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def localDensityAtI(cloud, x_k):
    M_i = len(cloud)
    sum_val = sum(d2(point, x_k) for point in cloud)
    ld = 1 / (1 + (1 / M_i) * sum_val)
    return ld

def globalDensity(clouds, x_k):
    k = sum(len(cloud) for cloud in clouds)
    sum_val = sum(d2(point, x_k) for cloud in clouds for point in cloud)
    gd = 1 / (1 + (1 / (k - 1) * sum_val))
    return gd

def membershipAtCloudI(cloudI, clouds, x_k):
    ld_i = localDensityAtI(cloudI, x_k)
    sum_val = sum(localDensityAtI(cloud, x_k) for cloud in clouds)
    memb = ld_i / sum_val
    return memb

def main():
    cloud1 = [(-1, -2), (-5, -10), (-12, -24), (-17, -34), (-20, -40)]
    cloud2 = [(0, 0), (5, 5), (12, 12), (17, 17), (20, 20)]

    clouds = [cloud1, cloud2]

    input_x, input_y = map(float, input("Enter input (x y): ").split())
    input_point = (input_x, input_y)

    i = 1
    for cloud in clouds:
        ld = localDensityAtI(cloud, input_point)
        print(f"Local density at cloud {i}: {ld}")

        memb = membershipAtCloudI(cloud, clouds, input_point)
        print(f"Membership at cloud {i}: {memb}\n")
        i += 1

    gd = globalDensity(clouds, input_point)
    print(f"Global density: {gd}\n")

if __name__ == "__main__":
    main()


# In[ ]:




