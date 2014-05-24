
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

###plot the data
plt.xlim( (gminX, gmaxX) )
plt.ylim( (gminY, gmaxY) )
plt.plot(gdiamond[:,:1], gdiamond[:, 1:2], '.')
plt.plot(grectangle[:,:1], grectangle[:, 1:2], '1')
plt.plot(gtriangle[:,:1], gtriangle[:, 1:2], '+')
plt.plot(gstar[:,:1], gstar[:, 1:2], '*')

################
"""
    Here we use cyclic_pla to binary classify two class.
"""
def cyclic_pla(td):
    x0 = np.zeros( (len(td), 1) )
    x0[:]=1.0
    td = np.concatenate( (x0, td[:,:1], td[:,1:2], td[:,2:3]), 1 )

    #The this initial value of w. td[0] include y. so we need to minus 1
    w=np.zeros( len(td[0])-1 );

    #
    #ensure all point corret
    stage=0;
    while(True):
        stage = stage+1;  
        #print("stage "+str(stage), w );
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
                #print("In stage "+str(stage)+".we need to update w ", w);
                print(idx, ty, sy)
                w = w + sy*sx
                isModifing = True
            #end if
        #end for
        
        print("The w is ", w)
      
        if(isModifing==False):
            break;
    #end while
    return w
#end


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
    weighOfPocketThres=0.05

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
            #print("The w is ", w, sy, sx)
            weight = calWeight(w, td)
            #if the new w is better than stuff in pocket, then update stuff in pocket
            if(weight<weighOfPocket):
                weighOfPocket = weight
                wPocket = w
            #end if        
            if(weighOfPocket<weighOfPocketThres):
                break;
        #end if       

        #print("The curIter is ", curIter)
        print("The weighOfPocket is ", weighOfPocket)
        print("The w is ", w)
        #drawSepLine(w, gminX, gmaxX)
    #end while
    return wPocket;
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

"""
Use the one versus all to calculate all w. store all w in ws
"""
def oneVersusAllHard(td, ws):
    pass;
    labels=[1,2,3,4] #we can get shi from rtd[:,2:3], we just skip this here
    for label in labels:
        nrtd = formOneVesusAll(td, label);
        #w=cyclic_pla(nrtd) #not work, since the nrtd are not binary classification strictly!!
        w = pocket(nrtd)
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
oneVersusAllHard(gtd, gw);

#plot the line
for w in gw:
    print("w :", w)
    drawSepLine(w, gminX, gmaxX)
#end for

#gw   : 1, 2, 3, 4
#lable: 1, 2, 3, 4

#plot test data
plt.plot(grtestData[:,:1], grtestData[:, 1:2], '_')

#update the test data
xt0 = np.zeros( (len(grtestData), 1) )
xt0[:]=1.0
gtestData = np.concatenate( (xt0, grtestData[:,:1], grtestData[:,1:2], grtestData[:,2:3]), 1 )

#test 
for e in gtestData:
    x = e[:len(e)-1]; y=e[len(e)-1]    
    msg = "For "+str(x)+" expented label:"+str(y)+", actual:"
    for w in gw:
        actualY=genLabel(x, w)
        msg += str(actualY) + ";";
    #end for
    print(msg)
#end for

pass


################
