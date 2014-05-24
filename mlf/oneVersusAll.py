
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


##############################################

#diamond
gdiamond=np.array([
[1.0, 12.0,   1.],
[1.5, 12.5,  1.], 
[3.5, 11.5,   1.], 
[4.5, 14.0,  1.],
[5.5, 16.0,  1.],
[6.0, 11.5,  1.],
[7.0, 10.5,  1.]
])

#rectangle
grectangle=np.array([
[9.5, 13.0,   2.],
[10.0, 11.5,  2.], 
[10.5, 11.5,  2.], 
[11.0, 13.0,  2.],
[12.0, 12.0,  2.],
[12.5, 12.5,  2.],
[13.0, 11.0,  2.], 
[14.0, 10.0,  2.], 
[15.0, 10.5,  2.], 
[15.5, 10.6,  2.]
])

#triangle
gtriangle=np.array([
[1.0, 2.5,   3.],
[2.0, 6.0,   3.],
[3.0, 2.0,  3.], 
[3.0, 5.0,   3.], 
[4.0, 2.2,  3.],
[4.0, 5.5,  3.],
[6.0, 2.0,  3.],
[6.0, 5.5,  3.],
[6.5, 2.0,  3.],
[6.7, 0.5,  3.]
])

#star
gstar=np.array([
[9.5, 8.5,   4.],
[10.0, 1.5,  4.], 
[11.0, 6.0,  4.],
[7.7, 6.0,  4.],
[8.0, 4.5,  4.],
[8.2, 4.0,  4.],
[9.0, 1.5,  4.],
[9.0, 4.5,  4.], 
[9.5, 5.0,  4.], 
[11.0, 1.5,  4.], 
])

grtd = np.concatenate((gdiamond,grectangle, gtriangle, gstar))
gminX = (int)(np.min(grtd[:,:1]))-3
gmaxX = (int)(np.max(grtd[:,:1]))+3
gminY = np.min(grtd[:,1:2])-3
gmaxY = np.max(grtd[:,1:2])+3

grtestData = np.array([
[15.0, 15.0,   2.],
[13.0, 4.0,  4.], 
[8.0, 8.0,  0.],
[10.0, 9.0,  0.],
[1.5, 7.0,  13.],
[2.0, 6.0,  13.],
[16.0, 7.0,  24.],
])

###plot the training data
plt.xlim( (gminX, gmaxX) )
plt.ylim( (gminY, gmaxY) )
plt.plot(gdiamond[:,:1], gdiamond[:, 1:2], '.')
plt.plot(grectangle[:,:1], grectangle[:, 1:2], '1')
plt.plot(gtriangle[:,:1], gtriangle[:, 1:2], '+')
plt.plot(gstar[:,:1], gstar[:, 1:2], '*')

################
"""
In many case, sy*np.inner(w, sx)--->+infite number-->thetav--->sum will not update-->grad will not update
How to fix this?
Here we use loop to ...
"""
def gradient_typical(td, w):
    sum = 0.
    num = len(td)
    for idx in range(num):
        sample = td[idx]
        sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]
        thetav = theta(-sy*np.inner(w, sx));
        #print("thetav is ", -sy*np.inner(w, sx), thetav)
        sum += thetav*(-sy*sx)         
    #end for
    #print("The is ", w, sum)
    gradAvg = sum/num
    return gradAvg
#end

"""
 Use random number to ...
"""
def gradient_rand(td, w):
    num = len(td)
    idx = np.random.randint(0, num)
    sample = td[idx]
    sx = sample[:len(sample)-1]; sy=sample[len(sample)-1]
    thetav = theta(-sy*np.inner(w, sx));
    gradAvg = thetav*(-sy*sx)         
    return gradAvg
#end

"""
    Here we use xxx to binary classify two class softly.
"""
def sgd(td):
    maxIter=1000000
    maxIter=500000
    maxIter=1000
    maxIter=10000
    maxIter=100000

    eta=0.0005
    eta=0.005
    eta=0.05

    #The this initial value of w. td[0] include y. so we need to minus 1
    w=np.zeros( len(td[0])-1 );
    #generate gradient threshold
    grad = gradient_rand(td, w)
    gradThres = np.zeros(grad.shape)
    gradThres[:]=0.0001

    curIter=0
    while(curIter<maxIter):
        curIter = curIter +1;
        #print("The curIter is ", curIter)
        
        grad = gradient_rand(td, w)
        tmp= np.less_equal( np.abs(grad), gradThres)
        if(np.all(tmp)):
            #Now we get a w that may be work, check this ...
            newGrad=gradient_typical(td, w)
            newTmp= np.less_equal( np.abs(newGrad), gradThres)
            if(np.all(newTmp)):
                break;
        #end if
        
        #print("The grad is ", grad)
        #print("The w is ", w)    
        #drawSepLine(w, minX, maxX);
        
        w = w - eta * grad    
    #end while
    return w
#end


################
"""
if the y in each element of nrtd is not equal to label, 
then set it as -1, thus we form the train data as one versus all.

Note:should set as -1 rather than 0!!!! refer to our current formula.
"""
def formOneVesusAll(td, label):
    ntd = td.copy()
    labelIdx = len(ntd[0])-1
    for e in ntd:
        if(e[labelIdx]!=label):
            e[labelIdx]=-1 #IMPORTANT
        else:
            e[labelIdx]=1  #IMPORTANT    
    #end
    return ntd
#end

labels=[1,2,3,4] #we can get shi from rtd[:,2:3], we just skip this here
glabels = labels
"""
Use the one versus all to calculate all w. store all w in ws
"""
def oneVersusAll(td, ws):
    pass;
    for label in labels:
        nrtd = formOneVesusAll(td, label);
        w = sgd(nrtd)
        ws.append(w)
        print("w for label ", label, " is ", w)
        pass;
    #end for
#end 

################
#add constant two the training data
x0 = np.zeros( (len(grtd), 1) )
x0[:]=1.0
gtd = np.concatenate( (x0, grtd[:,:1], grtd[:,1:2], grtd[:,2:3]), 1 )


gw=[];
oneVersusAll(gtd, gw);

#plot the line
for w in gw:
    print("w :", w)
    drawSepLine(w, gminX, gmaxX)
#end for

#gw   : 1, 2, 3, 4
#label: 1, 2, 3, 4
#probability: 

#plot test data
plt.plot(grtestData[:,:1], grtestData[:, 1:2], '_')

#update the test data
xt0 = np.zeros( (len(grtestData), 1) )
xt0[:]=1.0
gtestData = np.concatenate( (xt0, grtestData[:,:1], grtestData[:,1:2], grtestData[:,2:3]), 1 )

#is there any data structure like gp here.
gp=[];
#test 
for e in gtestData:
    x = e[:len(e)-1]; y=e[len(e)-1]    
    msg = "For "+str(x)+" expented label:"+str(y)+", actual:"
    ps=[];
    for idx in range(len(gw)):
        w = gw[idx]
        label = glabels[idx]
        probability=genProbability(x, w)
        ps.append( (label, probability) )
        msg += str(probability) + ";";
    #end for
    gp.append( (e, ps) )
    print(msg)
#end for

#print final result for test data.
for e in gp:
    key = e[0]
    values = e[1]
    midx=0; 
    for idx in range(len(values)):
        if(values[idx][1]>values[midx][1]):
            midx = idx
        #end if 
    #end for
    print(key, ", (label, p)=", values[midx])
#end for


################
