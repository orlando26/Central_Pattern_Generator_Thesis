
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

#distribution vector function
def vDis(e,vi):
    m = vi.size
    vo = np.full((1,m), e, dtype=float)
    return vo


# In[2]:

#distribution matrix function
def mDis(e, mi):
    m, n = mi.shape
    mo = np.full((m,n), e, dtype=float)
    return mo


# In[13]:

#gauss funcion
def gauss(x, cm, l):
    y = np.exp((-(x-cm)**2) /l)
    return y

#gaussian function for all the elements of the matrix
def EVFGauss(mi, cm, l):
    m, n = mi.shape
    mo = np.zeros((m, n), dtype=float)

    for i in range(0, m):
        for j in range(0, n):
            mo[i, j] = gauss(mi[i, j], cm, l)

    return mo


# In[4]:

#Chaotic gaussian neuron
def nGauss(sa):
    l = 0.15
    cm = 0.25
    sa = gauss(sa, cm, l)
    return sa


# In[5]:

#Artificial Neural System Single Output
def aNSSOutput(e, sa):
    w1 = np.array(([0.1, 0.2, 0.3],[0.6, 0.5, 0.4], [0.7, 0.8, 0.9]), dtype=float)
    w2 = np.array(([0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 0.9]), dtype=float)
    A = np.array(([0.1, 0.1, 0.1],[0.10, 1, 0.1], [0.1, 0, 0.1]), dtype=float)

    mAux = np.zeros((3,3), dtype=float)

    E = mDis(e, mAux)
    m1 = w1 - E

    sa = nGauss(sa)
    SA = mDis(sa, mAux)
    m2 = w2 - SA

    mAux = m1 + m2
    R = EVFGauss(mAux, 0.0, 0.15)

    mTemp = R * A

    s1 = np.sum(mTemp)
    s2 = np.sum(R)
    y = s1/(s2 + 0.00000052)

    return y, sa


# In[24]:

#Artificial Neural System Matrix Output
def aNSMOutput(e, sa):
    w1 = np.array(([0.1, 0.2, 0.3],[0.6, 0.5, 0.4], [0.7, 0.8, 0.9]), dtype=float)
    w2 = np.array(([0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.7, 0.8, 0.9]), dtype=float)
    A = np.array(([180, 1, 1],[1, 180, 1], [1, 180, 180]), dtype=float)

    mAux = np.zeros((3, 3), dtype=float)

    E = mDis(e, mAux)
    m1 = w1 - E

    sa = nGauss(sa)
    SA = mDis(sa, mAux)
    m2 = w2 - SA

    mAux = m1 + m2
    R = EVFGauss(mAux, 0.0, 0.15)

    mo = R * A

    return mo, sa


# In[22]:

#plots a group of 251 samples of the Neural System Output for a given stimulus(e)
def plotPattern1(e):
    sa = 1

    data = []
    for i in range(1, 251):
        mo, sa = aNSMOutput(e, sa)
        data.append(mo[1][1])
    return data


# In[23]:

#plots a group of 251 samples of the Neural System Output for a given stimulus(e)
def plotPattern(e, figure, subplot, color):
    sa = 0

    data = []
    for i in range(1, 251):
        mo, sa = aNSMOutput(e, sa)
        data.append(mo[1][1])

    plt.figure(figure)
    plt.subplot(subplot)
    plt.plot(data, color)
    plt.ylabel('Neuron (1,1) output.')
    plt.xlabel('n.')
    plt.title('Plot Pattern On e = ' + str(e))
    return data


# In[18]:

def getNeuralResponse(e):
    data = plotPattern1(e)
    xk = data[50:250]
    xkmo = data[49:249]
    return xk, xkmo

