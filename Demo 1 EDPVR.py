"""
Physics-based pressure-volume relationship (PBPVR) model for heart research.
Date: 2023-02-22
Author: Yunxiao Zhang, Moritz Kalhöfer-Köchling, Eberhard Bodenschatz, and Yong Wang
Email: yunxiao.zhang@ds.mpg.de; yong.wang@ds.mpg.de
Cite it: 
"""

from scipy.optimize import curve_fit
from pbpvr_lib import *
import matplotlib.pyplot as plt
import numpy as np
import os

# Input Parameters

# 1 Sarcomere Parameter
lamd_0=1.58/1.85

# 2 Normalized Thickness
D = 0.27

# 3 Passive Property
a = 1 #[kPa]
a = kpa_to_mmhg(a)
b = 3.8

# 4 Contractity
Ta = 50 #[kPa]
Ta = kpa_to_mmhg(Ta)

# 5 Ratio of passive and active parts
# "r_active = 0 yieds EDPVR"
# "r_passive = 1, r_active = 1 yieds ESPVR"
r_passive = 1
r_active = 0


# Input Volumes
vn2_esv = np.arange(1,2,0.01)
p= [PBPVR(v,D,a,b,Ta,r_passive,r_active,lamd_0) for v in vn2_esv] #[mmHg]


# Plot
plt.plot(vn2_esv,p,lw=4,color=colors[0],label=f"ESPVR")
plt.axvline(1,color='k')
plt.axhline(0,color='k')
plt.xlabel(r"Normalized Volume ($V/V_0$)")
plt.ylabel(r"Pressure ($mmHg$)")
plt.legend(loc="upper right")


plt.savefig("Demo 1 EDPVR generated from PBPVR model")
plt.show()

