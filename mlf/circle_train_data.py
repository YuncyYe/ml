
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math

##############################################
def sepLine(w, x):
    return -((w[0]+w[1]*x)/w[2])
#end

def drawSepLine(w, minX, maxX):
    sepx = range(minX, maxX)    
    sepy = []
    for e in sepx:
        tmp = sepLine(w, e)
        sepy.append( tmp )
    #end for
    plt.plot(sepx, sepy )
#end drawSepLine


##############################################
"""
"""
#1/(1+exp(-x))
def theta(x):
    ret = 0.
    try:
        ret = 1.0/( 1+math.exp(-x) )
    #except OverflowError:
    except:
        ret = 0.
        pass
    #end try
    return ret
#end

#hypothesis
def h(w, x):
    return theta(np.inner(w,x))
#end

"""
    Get the probability.
    Note: the x has the constant item.
    The return is 1 or -1.
    if 1, then the x belong to w.
"""
def genProbability(x, w):
    return h(w, x);
#emd

"""
    Get the label.
    Note: the x has the constant item.
    The return is 1 or -1.
    if 1, then the x belong to w.
"""
def genLabel(x, w):
    t = np.inner(w, x)
    ty = np.sign(t)
    return ty;
#emd

##############################################

nonCircle=np.array([
#left top
[1.2, 12.0,   -1.],
[1.3, 9.0,  -1.], 
[2.3, 11.5,   -1.], 
#left buttom
[0.8, 0.6,  -1.],
[1.5, 2.0,  -1.],
[1.8, 1.0,  -1.],
[2.3, 0.8,  -1.],
#right top
[8.9, 12.0,   -1.],
[12.0, 10.5,   -1.], 
#right buttom
[8.7, 1.9,  -1.],
[10.2, 1.4,  -1.]
])


circle=np.array([
[3.0, 8.0,   1.0],
[4.2, 5.5,  1.0], 
[5.0, 10.0,  1.0], 
[6.0, 6.8,  1.0],
[6.0, 5.0,  1.0],
[6.0, 4.9,  1.0],
[6.5, 7.9,  1.0], 
[6.9, 9.4,  1.0], 
[7.0, 4.2,  1.0], 
])


grtestData = np.array([
[1.2, 11.0,   -1.],
[1.2, 2.0,   -1.],
[11.0, 11.0,  -1.], 
[11.0, 3.0,  -1.],
[5.0, 5.0,  1.],
[6.5, 8.0,  1.],
[7.3, 10.0,  1.]
])

######The above sample's coordinate are error!!!.
######
nonCircle=np.array([
#left top
[1.2-6.0, 12.0-6.0,   -1.],
[1.3-6.0, 9.0-6.0,  -1.], 
[2.3-6.0, 11.5-6.0,   -1.], 
#left buttom
[0.8-6.0, 0.6-6.0,  -1.],
[1.5-6.0, 2.0-6.0,  -1.],
[1.8-6.0, 1.0-6.0,  -1.],
[2.3-6.0, 0.8-6.0,  -1.],
#right top
[8.9-6.0, 12.0-6.0,   -1.],
[12.0-6.0, 10.5-6.0,   -1.], 
#right buttom
[8.7-6.0, 1.9-6.0,  -1.],
[10.2-6.0, 1.4-6.0,  -1.]
])


circle=np.array([
[3.0-6.0, 8.0-6.0,   1.0],
[4.2-6.0, 5.5-6.0,  1.0], 
[5.0-6.0, 10.0-6.0,  1.0], 
[6.0-6.0, 6.8-6.0,  1.0],
[6.0-6.0, 5.0-6.0,  1.0],
[6.0-6.0, 4.9-6.0,  1.0],
[6.5-6.0, 7.9-6.0,  1.0], 
[6.9-6.0, 9.4-6.0,  1.0], 
[7.0-6.0, 4.2-6.0,  1.0], 
])


grtestData = np.array([
[1.2-6.0, 11.0-6.0,   -1.],
[1.2-6.0, 2.0-6.0,   -1.],
[11.0-6.0, 11.0-6.0,  -1.], 
[11.0-6.0, 3.0-6.0,  -1.],
[5.0-6.0, 5.0-6.0,  1.],
[6.5-6.0, 8.0-6.0,  1.],
[7.3-6.0, 10.0-6.0,  1.]
])


grtd = np.concatenate((nonCircle, circle))
gminX = (int)(np.min(grtd[:,:1]))-3
gmaxX = (int)(np.max(grtd[:,:1]))+3
gminY = np.min(grtd[:,1:2])-3
gmaxY = np.max(grtd[:,1:2])+3


###plot the training data
plt.xlim( (gminX, gmaxX) )
plt.ylim( (gminY, gmaxY) )
plt.plot(nonCircle[:,:1], nonCircle[:, 1:2], '.')
plt.plot(circle[:,:1], circle[:, 1:2], '*')



###two 
#X**2 -- >
nCC = nonCircle.copy()
for e in nCC:
    e[0] = math.pow(e[0], 2)
    e[1] = math.pow(e[1], 2)
#end for

CC = circle.copy();
for e in CC:
    e[0] = math.pow(e[0], 2)
    e[1] = math.pow(e[1], 2)
#end for

plt.plot(nCC[:,:1], nCC[:, 1:2], '*')
plt.plot(CC[:,:1], CC[:, 1:2], '+')

grtd2 = np.concatenate((nCC, CC))
gminX2 = (int)(np.min(grtd2[:,:1]))-3
gmaxX2 = (int)(np.max(grtd2[:,:1]))+3
gminY2 = np.min(grtd2[:,1:2])-3
gmaxY2 = np.max(grtd2[:,1:2])+3

nf = plt.figure()
plt.xlim( (gminX2, gmaxX2) )
plt.ylim( (gminY2, gmaxY2) )
plt.plot(nCC[:,:1], nCC[:, 1:2], '^')
plt.plot(CC[:,:1], CC[:, 1:2], '+')



###