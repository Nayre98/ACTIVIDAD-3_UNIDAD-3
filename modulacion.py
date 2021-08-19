#-*- coding: utf-8 -*-
"""
Created on Thu Now 5 17:59:37 2020
@author: Nayre
"""
#BIBLIOTECAS ESCOGIDAS PARA EL PROGRAMA

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import pylab as pl
from math import pi
from math import log10
plt.close('all')

#VARIABLES INGRESADAS POR EL USUARIO

vm= float(input('ingrese el valor de Amplitud de la Muduladora -->'))
Fm= int(input('Ingrese el valor de Frecuencia de la Moduladora -->'))
Vc= float(input('Ingrese el valor de la Amplitud de la portadora -->'))
Fc= int(input('Ingrese el valor de la Frecuencia de la Portadora -->'))
Kf= float(input('Ingrese el factor de sensibilidad de frecuencia -->'))
n= float(input('Ingrese el numero de periodos -->'))
print()

#VARIABLES QUE USAREMOS A LO LARGO DEL PROGRAMA

Z = 50
Δf=Kf*vm
β=Δf/Fm

Fs=50000
x=0
n0=[]
besse1=[]
f=np.arange(0,10,1)

#ECUACIONES PARA ALLAR BESSEL

for i in range(0,len(f)):
    x= round(sp.jv(i,β),2)
    besse1.append(x)
n_positivos=besse1[1:11];
n_negativos=np.flip(n_positivos);
n0.append(besse1[0]);
jn=np.concatenate((n_negativos,n0,n_positivos))
nB=4
BWc=2*Fm*nB
BWc=2*(Δf*vm)

#VALORES PARA LAS FRECUENCIAS
f_ns=[]
f_ps=[]
F0=[]
F0.append(Fc)
for f_inicial in range(0,len(f)):
    if f_inicial==0:
        f_1=Fc-Fm;
        f_inicial=f_1;
    else:
        f_1=f_1-Fm;
        f_inicial=f_1;
    f_ns.append(f_inicial);
finv_ns=np.flip(f_ns);
for f_final in range(0,len(f)):
    if f_final==0:
        f_1=Fc+Fm;
        f_final=f_1;
    else:
        f_1=f_1+Fm;
        f_final = f_1;
    f_ps.append(f_final);
finv_ps=np.flip(f_ps);
Fn = np.concatenate((finv_ns,F0,f_ps))
t=np.arange(0,n*1/Fm,1/Fs)


#HALLAR Vc * Jn
f_VcJn=[]
VcJn = 0
VcJn = np.round(abs((jn*Vc)/(np.sqrt(2))),2)
f_VcJn.append(VcJn)

#HALLAR VALORES EN dB DE Jn * Vc
f_VndB=[]
VndB=0
VndB=np.round(abs((20*np.log10(VcJn))),2)
f_VndB.append(VndB)

#HALLAR POTENCIA EN WATTS (W)
f_PnW=[]
PnW=0
PnW=abs(((jn*Vc)**2)/100)
f_PnW.append(PnW)

#HALLAR POTENCIA EN dBm
f_PndBm=[]
PndBm=0
PndBm=np.round(abs((10*np.log10(PnW*1000))),2)
f_PndBm.append(PndBm)

#CALCULO DE ECUACIONES PRESENTES EN EL PROGRAMA DE MODULACION
Vportadora=Vc*np.cos(2*pi*Fc*t);
Vmoduladora=vm*np.sin(2*pi*Fm*t);
Vfm=Vc*np.cos(2*pi*Fc*t+β*np.sin(2*pi*Fm*t));

#Formulas Resultantes
print('RESULTADOS MUDULACION FM')
print()
print('{:^10} {:^10} {:^10} {:^10}'.format('Δf','β','BWb','BWc'))
print('{:^10} {:^10} {:^10} {:^10}'.format(Δf,β,BWc,BWc))
print()
print('{:^10} {:^9} {:^9} {:^9} {:^9}'.format('Jn','Fn','Vc*Jn','Vn(dB)','Vn(dBm)','Vn(dBm)'))
for formatted in map('{:^10}{:^10}{:^9}{:^10}{:^10}'.format, jn,Fn,VcJn,VndB,PndBm):
      print(formatted)
print()
print("LA ECUACION PORTADORA ES:")
print("Vc(t)=",Vc,"cos(2π",Fc,"t)")
print()
print("LA ECUACION MODULADORA ES:")
print("Vm(t)=",vm,"sen(2π",Fm,"t)")
print()
print("LA ECUACION GENERAL PARA FM ES:")
print("Vfm(t)=",Vc,"cos[2π",Fc,"t +",β, "sen(2π",Fm,"t)]")
print()

#GRAFICA RESULTANTES
fig=plt.figure()
fig,plt.subplot(1,1,1)
plt.plot(t,Vportadora,color="blue",linewidth=0.8)
plt.title('señal portadora')
plt.xlabel('tiempo');
plt.ylabel('amplitud');
plt.grid(True)

fig1=plt.figure()
fig,plt.subplot(1,1,1)
plt.plot(t,Vportadora,color="black",linewidth=0.8)
plt.title('señal Moduladora')
plt.xlabel('tiempo');
plt.ylabel('amplitud');
plt.grid(True)

fig2=plt.figure()
fig,plt.subplot(1,1,1)
plt.plot(t,Vportadora,color="red",linewidth=0.8)
plt.title('Modulacion de Frecuencia')
plt.xlabel('tiempo');
plt.ylabel('amplitud');
plt.grid(True)
