
##
##Logistic Regression Algorithm
##
##Look out! I use some a lot of time in this model.
##Since i make mistake in the theta function implementation, h function implementation and the train data!
##
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math

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

##############################################
def sepLine(w, x):
    return h(w, x)
#end

def sepLinearLine(w, x):
    return -((w[0]+w[1]*x)/w[2])
#end

def drawSepLine(w, minX, maxX):
    sepx = range(minX, maxX)    
    sepy = []
    for e in sepx:
        #tmp = sepLine(w, [1,e])
        tmp = sepLinearLine(w, [1,e])
        #print(tmp)
        sepy.append( tmp )
    #end for
    plt.plot(sepx, sepy )
    #print(sepx, sepy)
#end drawSepLine
##############################################

#the y's value is like binary classification
########train data 1
rtd=np.array([
 [0.5,  0.]
,[1.0,  0.] 
,[2.5,  0.] 
,[4.0,  0.] 
,[4.5,  0.]
,[6.0,  0.]
,[7.0,  1.0]
,[8.0,  1.0]
,[9.0,  1.0]
,[10.0, 1.0]
,[11.0, 1.0]
,[12.0, 1.0]
,[13.0, 1.0]
])

testRData=np.array([
  [3.5, 0.]
 ,[5.5, 0.]
 ,[10.5, 1.]
 ,[11.5, 1.]
])

########train data 2
#The following data have bug
ls=np.array([
[1.0, 0.5,   1.],
[1.5, 14.5,  1.], 
[2.5, 1.5,   1.], 
[2.8, 3.5,  1.],
[4.5, 13.0,  1.],
[6.0, 8.0,  1.],
#[7.0, 16.0,  1.], #noize data
[8.0, 5.5,  1.],
[9.5, 7.0,  1.],
[12.0, 2.5,  1.],
[14.0, 2.0,  1.]
])
rs=np.array([
[2.0, 18.0, -1.],
[3.0, 17.5, -1.], 
#[3.5, -1.7, -1.], #noize data 
[8.0,11.5, -1.], 
[8.5,13.5, -1.], 
[8.5,13.0, -1.], 
[9.0,15, -1.], 
[12.0,20.0, -1.],
[16.0,17.0, -1.]
])

########train data 3
ls=np.array([
[1.0, 0.5,   1.],
[1.5, 14.5,  1.], 
[2.5, 1.5,   1.], 
[2.8, 3.5,  1.],
[4.5, 13.0,  1.],
[6.0, 8.0,  1.],
#[7.0, 16.0,  1.], #noize data
[8.0, 5.5,  1.],
[9.5, 7.0,  1.],
[12.0, 2.5,  1.],
[14.0, 2.0,  1.]
])
rs=np.array([
[2.0, 18.0, -1.],
[3.0, 17.5, -1.], 
#[3.5, -1.7, -1.], #noize data 
[8.0,11.5, -1.], 
[8.5,13.5, -1.], 
[8.5,13.0, -1.], 
[9.0,15, -1.], 
[12.0,20.0, -1.],
[16.0,17.0, -1.]
])

rtd = np.concatenate((ls,rs))
minX = (int)(np.min(rtd[:,:1]))-3
maxX = (int)(np.max(rtd[:,:1]))+3
minY = np.min(rtd[:,1:2])-3
maxY = np.max(rtd[:,1:2])+3

testRData = np.array([
[5.5, 5.0, 1.]
,[9.0, 3., 1.]
,[12., 2.4, 1.] #almost
,[5.5, 17.0, -1.]
,[11.0, 14., -1.]
,[12.0, 4.0, -1.] #almost
])

###plot the data
plt.xlim( (minX, maxX) )
plt.ylim( (minY, maxY) )
plt.plot(ls[:,:1], ls[:, 1:2], '*')
plt.plot(rs[:,:1], rs[:, 1:2], '+')

##construct training data
x0 = np.zeros( (len(rtd), 1) )
x0[:]=1.0
td = np.concatenate( (x0, rtd[:,:1], rtd[:,1:2], rtd[:,2:3]), 1 )

#The this initial value of w. td[0] include y. so we need to minus 1
w=np.zeros( len(td[0])-1 );

"""
In many case, sy*np.inner(w, sx)--->+infite number-->thetav--->sum will not update-->grad will not update
How to fix this?
"""
def gradient(td, w):
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
    return sum/num
#end

##############logra-begin
maxIter=1000000
maxIter=500000
maxIter=1000
maxIter=10000
maxIter=100000

eta=0.0005
eta=0.005
eta=0.05

#generate gradient threshold
grad = gradient(td, w)
gradThres = np.zeros(grad.shape)
gradThres[:]=0.0001

curIter=0
while(curIter<maxIter):
    curIter = curIter +1;
    #print("The curIter is ", curIter)
    
    grad = gradient(td, w)
    tmp= np.less_equal( np.abs(grad), gradThres)
    if(np.all(tmp)):
        break;
    #end if
    
    print("The grad is ", grad)
    #print("The w is ", w)    
    #drawSepLine(w, minX, maxX);
    
    w = w - eta * grad    
#end

##############logra-end

print("The curIter is ", curIter)
print("The grad is ", grad)
print("The w is ", w)

#show the seperator line
#drawSepLine(w, minX, maxX);

#
#test data
testDataX = np.concatenate((testRData[:,0:1], testRData[:,1:2]), 1)
testDataExpY = testRData[:,2:3]
testDataY = []
for idx in range(len(testDataX)):
    e = testDataX[idx]
    tmp = sepLine(w, [1,e[0],e[1]])
    testDataY.append( tmp )
    print(e, tmp, testDataExpY[idx])
    pass
#end for
#plt.plot(testDataX, testDataY, '.' )


###
#E:\cloud_storage\kuaipan\course\python_ml_trying>ipython --matplot
##...
#In [93]: import LogRA
#In [94]: reload(LogRA)
#
if __name__ =="__main__":
    pass
#end


