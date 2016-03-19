import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
from pylab import *

def MatrixCreate(x,y):
    return np.zeros((x, y))

def VectorCreate(x):
    return np.zeros((x),dtype='f')

def random():
    return np.random.uniform()
def MatrixRandomize(v):
    for (x,y) in np.ndindex(v.shape):
        v[x, y]=2*random()-1
    return v

def MatrixPerturb(p,a):
    q=p.copy()
    for (x,y) in np.ndindex(q.shape):
        if(random()<a):
            q[x,y]=2*random()-1
        else:
            q[x,y]=p[x,y]
    return q

numUpdates=10
numNeuros=10

def Update(v,s):
    for n in range(len(v[:,0])-1):
        for i in range(len(s[0,:])):
            temp=np.dot(v[n,:],s[:,i])
            if temp<0:
                temp=0
            elif temp>1:
                temp=1
            v[n+1,i]=temp
    return v

parent=MatrixCreate(10,10)
parent=MatrixRandomize(parent)

desiredNeuronValues = VectorCreate(10)
for j in range(1,10,2):
    desiredNeuronValues[j]=1
    
def MeanDistance(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2)/numNeuros)

def Fitness(v):
    return 1-MeanDistance(v[numUpdates-1,:],desiredNeuronValues)

fits=MatrixCreate(1,1000)
neuronValues=MatrixCreate(numUpdates,numNeuros)
neuronValues[0,:]=0.5
neuronValues=Update(neuronValues,parent)
pylab.imshow(neuronValues, cmap=cm.gray, aspect='auto', interpolation='nearest')
plt.savefig('a.pdf')
plt.show()
parentFitness = Fitness(neuronValues)

for currentGeneration in range(0,1000):
    print(currentGeneration, parentFitness)
    neuronValues=MatrixCreate(numUpdates,numNeuros)
    neuronValues[0,:]=0.5
    child = MatrixPerturb(parent,0.05)
    neuronValues=Update(neuronValues,child)
    childFitness = Fitness(neuronValues)
    fits[0,currentGeneration]=parentFitness
    if ( childFitness > parentFitness ):
        parent = child.copy()
        parentFitness = childFitness
        
def PlotVectorAsLine(v):
    plt.plot(v[0])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

pylab.imshow(neuronValues, cmap=cm.gray, aspect='auto', interpolation='nearest')
plt.savefig('b.pdf')
plt.show()
PlotVectorAsLine(fits)
plt.savefig('c.pdf')
plt.show()