

#
#pocket Algorithm 
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

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

ls=np.array([
[1.0, 0.5,   1]
,[1.5, 14.5,  1] 
,[2.5, 1.5,   1] 
,[2.8, 3.5,  1]
,[4.5, 13.0,  1]
,[6.0, 8.0,  1]
,[7.0, 16.0,  1] #noize data
,[8.0, 5.5,  1]
,[9.5, 7.0,  1]
,[12.0, 2.5,  1]
,[14.0, 2.0,  1]
#,[7.0, 16.0,  1] #noize data
])
rs=np.array([
[2.0, 18.0, -1]
,[3.0, 17.5, -1]
,[3.5, 0.7, -1] #noize data 
,[8.0,11.5, -1] 
,[8.5,13.5, -1] 
,[8.5,13.0, -1] 
,[9.0,15, -1]
,[12.0,20.0,-1]
,[16.0,17.0,-1]
#,[3.5, 0.7, -1] #noize data 
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

#todo:we can set it as max of float 
weighOfPocket=1000000000.0
wPocket=w

#
#ensure all point corret
#maxIter=900000
maxIter=1200000
weighOfPocketThres=0.05

curIter=0
while(curIter<maxIter):
    curIter = curIter +1;
    
    #[begin----the following is typical pla----
    isModifing=False;    
    #check each point for w
    for ti in range(len(td)):
        rndIdx=random.randint(0, len(td)-1)
        sample = td[rndIdx]
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
        #todo. we need to update pocket here.
    #end]
        
    #pick up an element in sample to try to improve w
    #rndIdx=random.randint(0, len(td)-1)
    #sample = td[rndIdx]
    #sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]
    #w = w + sy*sx
    
    #It's too late to check weight for this w
    #calc weight for w
    weight=0.;
    for idx in range(len(td)):
        sample = td[idx]
        sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]
        t = np.inner(w, sx)
        ty = np.sign(t)
        #print(idx, ty, sy)
        if(ty!=sy):
            weight += 1.0;
    #end for
    
    #print("The curIter is ", curIter)
    #print("The weighOfPocket is ", weighOfPocket)
    #print("The w is ", w)
    #drawSepLine(w, minX, maxX)
    
    #if the new w is better than stuff in pocket, then update stuff in pocket
    if(weight<weighOfPocket):
        weighOfPocket = weight
        wPocket = w
    #end if
    
    if(weighOfPocket<weighOfPocketThres):
        break;
#end for

##############pla-end
print("The curIter is ", curIter)
print("The weighOfPocket is ", weighOfPocket)
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




