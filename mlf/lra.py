

#
#lra: Linear Regression Algorithm
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

##############################################
def sepLine(w, x):
    return ((w[0]+w[1]*x))
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

rtd=np.array([
 [0.5, 14.0]
,[1.0, 9.0] 
,[4.0, 7.0] 
,[4.5, 12.0]
,[6.0, 7.3]
,[8.0, 1.5]
,[9.0, 4.0]
,[10.0, 3.5]
,[11.0, 4.5]
,[13.0, 5.0]
])

minX = (int)(np.min(rtd[:,:1]))-3
maxX = (int)(np.max(rtd[:,:1]))+3
minY = np.min(rtd[:,1:2])-3
maxY = np.max(rtd[:,1:2])+3

##construct training data
x0 = np.zeros( (len(rtd), 1) )
x0[:]=1.0
td = np.concatenate( (x0, rtd[:,:1], rtd[:,1:2]), 1 )

###plot the data
plt.xlim( (minX, maxX) )
plt.ylim( (minY, maxY) )
plt.plot(rtd[:,:1], rtd[:, 1:2], '*')

##############lra-begin
X = td[:, :2]
Y = td[:, 2:3]

#pseudo-inverse
Xi=np.linalg.pinv(X)
#w's type is matrix
w=np.matrix(Xi)*Y 
#w's type is array
w=np.array(w)

##############lra-end

print("The w is ", w)

#show the seperator line
drawSepLine(w, minX, maxX);



###
#In [93]: import pla
#In [94]: reload(pla)
#
if __name__ =="__main__":
    pass
#end





