"""
Physics-based pressure-volume relationship (PBPVR) model for heart research.
Date: 2023-02-22
Author: Yunxiao Zhang, Moritz Kalhöfer-Köchling, Eberhard Bodenschatz, and Yong Wang
Email: yunxiao.zhang@ds.mpg.de; yong.wang@ds.mpg.de
Cite it: 
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt
from scipy.integrate import quad
from scipy.optimize import root,minimize

# define units
from typing import NewType
mmHg = NewType('mmHg',float)
kPa = NewType('kPa',float)
Vn1 = NewType('Vn1',float)
Vn2 = NewType('Vn2',float) 
mL = NewType('mL',float)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# help functions
def kpa_to_mmhg(p:kPa):
    "convert kpa to mmhg"
    return 7.50062*np.array(p)
    
def mmhg_to_kpa(p:mmHg): 
    "convert mmhg to kpa"
    return np.array(p)/7.50062

def volume_of_sphere(r): 
    "return the volume of a sphere"
    return 4/3*np.pi*r**3

# "(03 file related functions)"
def skip_nlines(fp,n):
    for i in range(n): 
        fp.readline()

def read_2columns(fp,i,j,is_csv = False):
    a_list = []
    b_list = []
    for line in fp:
        if is_csv == False:
            line_list = line.strip().split()
        else:
            line_list = line.strip().split(",")
        a_list.append(float(line_list[i]))
        b_list.append(float(line_list[j]))
    return np.array(a_list),np.array(b_list)

def read_3columns(fp,i,j,k,is_csv = False):
    a_list = []
    b_list = []
    c_list = []
    for line in fp:
        if is_csv == False:
            line_list = line.strip().split()
        else:
            line_list = line.strip().split(",")
        a_list.append(float(line_list[i]))
        b_list.append(float(line_list[j]))
        c_list.append(float(line_list[k]))
    return np.array(a_list),np.array(b_list),np.array(c_list)


# "(04 statistic functions)"
def MAE(v1,v2):
    "(Mean Absolute Error)"
    return sum(abs(v1-v2))/len(v1)

def MSE(v1,v2):
    "(Mean Squared Error)"
    return sum((v1-v2)**2)/len(v1)

def RMSE(v1,v2):
    "(Root of Mean Squared Error)"
    return np.sqrt(sum((v1-v2)**2)/len(v1))

def SD(v_predicted,v):
    "(Standard Devation)"
    residuals =  (v_predicted-v)
    residuals_avg = sum(residuals)/len(residuals)
    return sqrt(sum((residuals - residuals_avg)**2)/(len(residuals)-1))
    
def corr(pcov):
    """
    return the correction of fitted parameter 
    based on pcov of curve_fitting from scipy
    """
    corr = np.zeros_like(pcov)
    n = len(pcov[0])
    for i in range(n):
        for j in range(n):
            corr[i,j] = pcov[i,j]/(sqrt(pcov[i,i])*sqrt(pcov[j,j]))
    return corr

def SE(v_predicted,v):
    """
    Standard Error of the Mean
    Standard Error
    """
    return SD(v_predicted,v)/sqrt(len(v))

def R_squared(v_predicted,v):
    """
    coefficient of determination
    R**2 = 1 - RSS/TSS
    """
    RSS = sum((v-v_predicted)**2)
    v_avg = sum(v)/len(v)
    TSS = sum((v-v_avg)**2)
    return 1-RSS/TSS

"""
two volume normalization methods:
Vn1: Klotz method, Vn1 = (V-V0)/(V30-V0)
Vn2: My mythod, Vn2 = V/V0
"""

# Volume Coversion
def V_to_Vn1(V:mL, V0, V30):
    Vn1 = (V-V0)/(V30-V0)
    return Vn1

def Vn1_to_V(Vn, V0, V30)->mL:
    Vn = np.array(Vn)
    V = (V30-V0)*Vn+V0
    return V

def V_to_Vn2(V, V0):
    Vn2 = V/V0
    return Vn2

def Vn2_to_Vn1(Vn2, V0, V30):
    Vn1 = V0*(Vn2-1)/(V30-V0)
    return Vn1

def Vn2_to_Vn1_2(Vn2, V30_V0):
    Vn1 = (Vn2-1)/(V30_V0-1)
    return Vn1

def Vn1_to_Vn2(Vn1, V0, V30):
    Vn1 = np.array(Vn1)
    Vn2 = (V30/V0-1)*Vn1+1
    return Vn2

def Vn1_to_Vn2_2(Vn1, V30_V0):
    Vn1 = np.array(Vn1)
    Vn2 = (V30_V0-1)*Vn1+1
    return Vn2


# Handle array

def find_index(v,v0):
    """
    find the index of value larger than v0 in list v
    v is increasing monotonously
    
    e.g.
    v = [0.1 0.23 0.2 0.21 0.3 0.35 ...]
    find the index of > 0.3
    return 5
    """
    for i,x in enumerate(v):
        if x>v0:
            return i
    print("No index fond")
    return len(v)

def find_near_value(x0,x,y):
    "x y known, given x0, find nearst x and return x,y pair"
    i = np.argmin([abs(x0-x_) for x_ in x])
    return x[i],y[i]

def to_equalx_data(x_want,x,y):
    "x y array known, give x_want, find nearst x,y pair to x_want"
    index = [np.argmin([abs(x0-x_) for x_ in x]) for x0 in x_want]
    x_new = [x[i] for i in index]
    y_new = [y[i] for i in index]
    return x_new,y_new

# EDPVR Famous Models
def klotz_curve(Vn1, A, B): 
    """
    Vn (1)
    A (mmHg)
    B (1)
    return p (mmHg)
    """
    Vn1 = np.array(Vn1)
    p = A*Vn1**B
    return p

def sunagawa_curve(V, V0, A, B):
    """
    V (mL)
    V0 (mL)
    A (kPa)
    B (1)
    return p (kPa)
    """
    V = np.array(V)
    p = A*(exp(B*(V-V0))-1)
    return p
