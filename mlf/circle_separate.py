


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math

"""
There are cicle to ..., but we're not assured to found it.


"""

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

def genLineSamples(w, minX, maxX):
    sepx = range(minX, maxX)    
    sepy = []
    for e in sepx:
        tmp = sepLine(w, e)
        sepy.append( tmp )
    #end for
    return sepx, sepy 
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

"""
feature transmation
We need to define this function according to you problem.
"""
def featureTransform(samples):
    newsamples = samples.copy()
    for e in nCC:
        e[0] = math.pow(e[0], 2)
        e[1] = math.pow(e[1], 2)
    #end for
    return newsamples
#end

    
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


###two 
#X**2 -- >
nCC = featureTransform(nonCircle)

CC = featureTransform(circle)

grtd2 = np.concatenate((nCC, CC))

################
"""
    Here we use pocket to binary classify two class.
"""
def pocket(td):
    #The this initial value of w. td[0] include y. so we need to minus 1
    w=np.zeros( len(td[0])-1 );

    #todo:we can set it as max of float 
    weighOfPocket=1000000000.0
    wPocket=w #w in pocket, that current best w.

    #
    #ensure all point corret
    maxIter=900000
    maxIter=1200000
    maxIter=42000
    #maxIter=22000
    weighOfPocketThres=0.0005

    #calc weight for w
    def calWeight(w, td):
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
        return weight;
    #end

    curIter=0
    while(curIter<maxIter):
        curIter = curIter +1;
        print("The curIter is ", curIter)

        #pick up an element in sample to try to improve w
        rndIdx=random.randint(0, len(td)-1)
        sample = td[rndIdx]
        sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]    
        t = np.inner(w, sx)
        ty = np.sign(t)
        print(rndIdx, ty, sy)
        if(ty!=sy):
            #failed, we need to update w
            w = w + sy*sx
            w [2] = w[1] #ensure it can be circle!
            #print("The w is ", w, sy, sx)
        #end if 
        weight = calWeight(w, td)
        #if the new w is better than stuff in pocket, then update stuff in pocket
        if(weight<weighOfPocket):
            weighOfPocket = weight
            wPocket = w
        #end if        
        print("The weighOfPocket is ", weighOfPocket)
        print("The w is ", w)
        
        if(weighOfPocket<weighOfPocketThres):
            break;

        #drawSepLine(w, gminX, gmaxX)
    #end while
    return wPocket;
#end 

######################

#add constant two the training data
x0 = np.zeros( (len(grtd2), 1) )
x0[:]=1.0
gtd = np.concatenate( (x0, grtd2[:,:1], grtd2[:,1:2], grtd2[:,2:3]), 1 )



w = pocket(gtd)



######################
fig0 = plt.figure(0)
ax0=fig0.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))

fig1 = plt.figure(1)
ax1=fig1.add_subplot(111)

###plot the training data
#ax0.xlim( (gminX, gmaxX) )
#ax0.ylim( (gminY, gmaxY) )
ax0.plot(nonCircle[:,:1], nonCircle[:, 1:2], '^')
ax0.plot(circle[:,:1], circle[:, 1:2], '+')


gminX2 = (int)(np.min(grtd2[:,:1]))-3
gmaxX2 = (int)(np.max(grtd2[:,:1]))+3
gminY2 = np.min(grtd2[:,1:2])-3
gmaxY2 = np.max(grtd2[:,1:2])+3


#ax1.xlim( (gminX2, gmaxX2) )
#ax1.ylim( (gminY2, gmaxY2) )
ax1.plot(nCC[:,:1], nCC[:, 1:2], '^')
ax1.plot(CC[:,:1], CC[:, 1:2], '+')

########################
tsepx, tsepy = genLineSamples(w, gminX2, gmaxX2)
ax1.plot(tsepx, tsepy)

#plot test data
ax0.plot(grtestData[:,:1], grtestData[:, 1:2], '*')


grtestData2 = featureTransform(grtestData)

#update the test data
xt0 = np.zeros( (len(grtestData2), 1) )
xt0[:]=1.0
gtestData = np.concatenate( (xt0, grtestData2[:,:1], grtestData2[:,1:2], grtestData2[:,2:3]), 1 )

for idx in range(len(gtestData)):
    e = gtestData[idx]
    x = e[:len(e)-1]; y=e[len(e)-1]
    y = genLabel(x, w)    
    msg = "For "+str( grtestData[idx] )+ " in circle " + str( y==1 )
    print(msg)
#end for


#rx1 = np.linspace(gminX, gmaxX, num=300)

#to prove that there are circle....
rx1 = np.linspace(-5, 5, num=300)
ry1 = []
ry2 = []
cw = [-20.0, 1., 1.0] #test
for e in rx1:
    tmp = (-cw[0] - cw[1]*e*e)/cw[2] 
    if(tmp<0):
        tmp=900
    tmp = math.sqrt(tmp)
    ry1.append( tmp )
    ry2.append( -tmp )
#end for
ax0.plot(rx1, ry1, '.')
ax0.plot(rx1, ry2, '.')


#Draw curve by w. This may not be circle. ....
rx1 = np.linspace(gminX, gmaxX, num=300)
ry1 = []
ry2 = []
for e in rx1:
    tmp = (-w[0] - w[1]*e*e)/w[2] 
    if(tmp<0):
        tmp=700
    tmp = math.sqrt(tmp)
    ry1.append( tmp )
    ry2.append( -tmp )
#end for
ax0.plot(rx1, ry1, '.')
ax0.plot(rx1, ry2, '.')

pass


###



