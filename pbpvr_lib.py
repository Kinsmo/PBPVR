"""
Libiary for Physics-based Pressure-Volume Relationship (PBPVR)
Date: 2023-02-22
Author: Yunxiao Zhang
Email: yunxiao9277@gmail.com
Cite it: 
"""

from pbpvr_help_functions import *

"(00 Basis)"

"(Normalized R)"
def Rn(d): 
    return 1+d

"(Normalized r)"
def rn(d,Vn): 
    return (Rn(d)**3+Vn-1)**(1/3)

"(radial strain lambda_rho)"
def lamd(d,Vn): 
    return Rn(d)**2/rn(d,Vn)**2

"(the first invariant of lambda_rho)"
def I1(lamd): 
    return (lamd**2+2/lamd)

"(02 Passive Part)"

"(P_ED part to integrate)"
def dW(d,Vn,a,b): 
    return  2*a * (1-lamd(d,Vn)**3)/rn(d,Vn) * exp(b*(I1(lamd(d,Vn))-3))

"(P_ED part for single input)"
def moritz_curve(Vn,D,a,b):
    # Vn = np.array(Vn)
	return quad(dW,0,D,args=(Vn,a,b))[0]

"(P_ED part for array input)"
def vmoritz_curve(Vn_list:Vn2,D,a,b):
    # Vn = np.array(Vn)
	return np.array([quad(dW,0,D,args=(Vn,a,b))[0] for Vn in Vn_list])

"(P_ED part for array input)"
def moritz_curve_for_fitting(Vn1_list:Vn1,D,a:mmHg,b):
    Vn2_list=Vn1_to_Vn2_2(Vn1_list,V30_V0=V30_V0(D,a,b))
    return np.array([quad(dW,0,D,args=(Vn2,a,b))[0] for Vn2 in Vn2_list]) #kPa

"(P_ED part for array: convert p to v)"
def vmoritz_curve_ptov(p:mmHg,D,a:mmHg,b)->Vn2:
    f = lambda Vn2: (vmoritz_curve(Vn2,D,a,b)-p)**2
    return minimize(f,1.6).x[0]

"(Find Normalized V30 of 2nd method)"
def V30_V0(D,a:mmHg,b)->Vn2:
    f = lambda Vn2: (vmoritz_curve(Vn2,D,a,b)-30)**2
    return minimize(f,1.6).x[0]


"(03 Active Part)"

"(P_a part to integrate)"
def dW_a(d,Ta,Vn,lamd_0=1.58/1.85):
    return 2*Ta * (1-lamd_0*sqrt(lamd(d,Vn)/2))/rn(d,Vn)

"(P_a part for single input)"
def yunxiao_active(Vn,D,Ta,lamd_0):
    return quad(dW_a,0,D,args=(Ta,Vn,lamd_0))[0]

"(04 Full PBPVR model with scaling factor, for for single input)"
def PBPVR(Vn,D,a,b,Ta,p_passive,p_active,lamd_0=1.58/1.85):
    p = p_passive * moritz_curve(Vn,D,a,b) + p_active * yunxiao_active(Vn,D,Ta,lamd_0)
    return p


if __name__ == "__main__":
    x = [1,2,3]
    y = [2,4,7]
    a = to_equalx_data([1,2,3,4,5,4,3,32,2],x,y)
    print(a)