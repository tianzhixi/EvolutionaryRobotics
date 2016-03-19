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
        v[x, y]=random()
    return v

def Fitness(v):
    return np.mean(v)

def MatrixPerturb(p,a):
    q=p.copy()
    for (x,y) in np.ndindex(q.shape):
        if(random()<a):
            q[x,y]=random()
        else:
            q[x,y]=p[x,y]
    return q

color=['b-','g-','r-','c-','m-']

def PlotVectorAsLine(v,c):
    plt.plot(v[0],c)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

Genes=MatrixCreate(50,5000)

for i in range(4):
    fits=MatrixCreate(1,5000)
    parent = MatrixCreate(1,50)
    parent = MatrixRandomize(parent)
    parentFitness = Fitness(parent)
    for currentGeneration in range(0,5000):
        print(currentGeneration, parentFitness)
        child = MatrixPerturb(parent,0.05)
        childFitness = Fitness(child)
        fits[0,currentGeneration]=parentFitness
        Genes[:,currentGeneration]=parent[0,:].transpose()
        if ( childFitness > parentFitness ):
            parent = child
            parentFitness = childFitness
    fig = PlotVectorAsLine(fits,color[i])

print(Genes)
plt.savefig('fitness.pdf')

pylab.imshow(Genes, cmap=cm.gray, aspect='auto', interpolation='nearest')
plt.xlabel('Generation')
plt.ylabel('Gene')
plt.savefig('Gene.pdf')