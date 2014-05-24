

#
#Perceptron Learning Algorithm - pla 
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

##sample 1
ls=np.array([
[3.,  9.5,   1],
[3.5, 3.0,  1], 
[4.0, 9., 1], 
[4.3, 1.0,  1]
])
rs=np.array([
[5.9,9.3, -1],
[6.,8.0, -1], 
[6., 2., -1], 
[7.0, 5.0, -1], 
[7.5,8.0,-1]
])


##sample 2
ls=np.array([
[3.5, 5.0,   1],
[4.0, 15.0,  1], 
[5.5, 13.,   1], 
[6.0, 9.0,  1],
[6.5, 11.0,  1],
[7.0, 8.0,  1],
[7.5, 13.0,  1],
[12.0, 18.0,  1],
[13.0, 14.0,  1]
])
rs=np.array([
[7.0, 5.0, -1],
[11., 6.0, -1], 
[13., 8.0, -1], 
[16.5,12.5, -1], 
[18.5,6.0,-1],
[20.0,7.0,-1]
])

##sample 3
ls=np.array([
[1.0, 0.5,   1],
[1.5, 14.5,  1], 
[2.5, 1.5,   1], 
[2.8, 3.5,  1],
[4.5, 13.0,  1],
[6.0, 8.0,  1],
#[7.0, 16.0,  1], #noize data
[8.0, 5.5,  1],
[9.5, 7.0,  1],
[12.0, 2.5,  1],
[14.0, 2.0,  1]
])
rs=np.array([
[2.0, 18.0, -1],
[3.0, 17.5, -1], 
#[3.5, 0.7, -1], #noize data 
[8.0,11.5, -1], 
[8.5,13.5, -1], 
[8.5,13.0, -1], 
[9.0,15, -1], 
[12.0,20.0,-1],
[16.0,17.0,-1]
])


##construct training data
rtd = np.concatenate((ls,rs))
minX = (int)(np.min(rtd[:,:1]))-3
maxX = (int)(np.max(rtd[:,:1]))+3

###plot the data
plt.xlim( (minX, maxX) )
plt.ylim( (np.min(rtd[:,1:2]-3), np.max(rtd[:,1:2]+3)) )
plt.plot(ls[:,:1], ls[:, 1:2], '*')
plt.plot(rs[:,:1], rs[:, 1:2], '+')

##############pla-begin
x0 = np.zeros( (len(rtd), 1) )
x0[:]=1.0
td = np.concatenate( (x0, rtd[:,:1], rtd[:,1:2], rtd[:,2:3]), 1 )

#The this initial value of w. td[0] include y. so we need to minus 1
w=np.zeros( len(td[0])-1 );

#
#ensure all point corret
stage=0;
while(True):
    stage = stage+1;  print("stage "+str(stage), w );
    pass
    
    isModifing=False;
    
    #check each point for w
    for idx in range(len(td)):
        sample = td[idx]
        sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]
        t = np.inner(w, sx)
        ty = np.sign(t)
        #print(idx, ty, sy)
        if(ty!=sy):
            #failed, we need to update w
            w = w + sy*sx
            isModifing = True
        #end if
    #end for
    
    if(isModifing==False):
        break;
#end while

##############pla-end

print("The w is ", w)


#show the seperator line

def sepLine(x):
    return -((w[0]+w[1]*x)/w[2])
#end

sepx = range(minX, maxX)    
sepy = []
for e in sepx:
    tmp = sepLine(e)
    sepy.append( tmp )
#end for
plt.plot(sepx, sepy )

###
#In [93]: import pla
#In [94]: reload(pla)
#
if __name__ =="__main__":
    pass
#end


