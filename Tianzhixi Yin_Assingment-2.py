import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
from pylab import *

def MatrixCreate(x,y):
    return np.zeros((x, y))

def random():
    return np.random.uniform()
def MatrixRandomize(v):
    for (x,y) in np.ndindex(v.shape):
        v[x, y]=2*random()-1
    return v


neuronValues=MatrixCreate(50,10)
for i in range(len(neuronValues[0,:])):
    neuronValues[0,i]=random()

neuronPositions=MatrixCreate(2,10)
numNeurons=len(neuronPositions[0,:])
angle = 0.0
angleUpdate = 2 * pi /numNeurons

for i in range(0,numNeurons):
    neuronPositions[0,i] = sin(angle)
    neuronPositions[1,i] = cos(angle)
    angle = angle + angleUpdate

plt.plot(neuronPositions[0,:],neuronPositions[1,:],'ko',markerfacecolor=[1,1,1], markersize=18)

plt.savefig('a.pdf')
plt.show()

synapses=MatrixCreate(10,10)
synapses=MatrixRandomize(synapses)
def plotfn_1(v):
    plt.plot(v[0,:],v[1,:],'ko',markerfacecolor=[1,1,1], markersize=18)
    for i in range(len(v[0,:])):
        for j in range(len(v[0,:])):
            plt.plot([v[0,i],v[0,j]],[v[1,i],v[1,j]])
    plt.savefig('b.pdf')
    plt.show()
plotfn_1(neuronPositions)

def plotfn_2(v,u):
    plt.plot(v[0,:],v[1,:],'ko',markerfacecolor=[1,1,1], markersize=18)
    for i in range(len(v[0,:])):
        for j in range(len(v[0,:])):
            if u[i,j]<0:
                plt.plot([v[0,i],v[0,j]],[v[1,i],v[1,j]],color=[0.8,0.8,0.8])
            else:
                plt.plot([v[0,i],v[0,j]],[v[1,i],v[1,j]],color=[0,0,0])
    plt.savefig('c.pdf')
    plt.show()
plotfn_2(neuronPositions,synapses)

def plotfn_3(v,u):
    plt.plot(v[0,:],v[1,:],'ko',markerfacecolor=[1,1,1], markersize=18)
    for i in range(len(v[0,:])):
        for j in range(len(v[0,:])):
            w = int(10*abs(synapses[i,j]))+1
            if u[i,j]<0:
                plt.plot([v[0,i],v[0,j]],[v[1,i],v[1,j]],color=[0.8,0.8,0.8],linewidth=w)
            else:
                plt.plot([v[0,i],v[0,j]],[v[1,i],v[1,j]],color=[0,0,0],linewidth=w)
    plt.savefig('d.pdf')
    plt.show()
plotfn_3(neuronPositions,synapses)

def Update(v,s):
    for n in range(len(v[:,0])-1):
        for i in range(len(s[0,:])):
            temp=np.dot(v[n,:],s[:,i])
            if temp<0:
                temp=0
            elif temp>1:
                temp=1
            v[n+1,i]=temp
            print(temp)
    return v

Neuro=Update(neuronValues,synapses)

pylab.imshow(Neuro, cmap=cm.gray, aspect='auto', interpolation='nearest')
plt.xlabel('Neuron')
plt.ylabel('Time Step')
plt.savefig('e.pdf')
plt.show()