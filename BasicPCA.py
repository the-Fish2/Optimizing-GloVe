import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets

def dataAnalysis(dim, data):
    pca = decomposition.PCA(n_components=dim)
    pca.fit(data)
    return pca.transform(data)

def visualize(dim, data):
    if dim == 3 or dim == 2:
        fig = plt.figure()
        plt.clf()

        if dim == 3:
            ax = fig.add_subplot(projection = "3d") #this is rectilinear, 3d, etc. projection= "3d"
        elif dim == 2:
            ax = fig.add_subplot(projection = "rectilinear")
        
        ax.set_position([0, 0, 0.95, 1])
        plt.cla()

        if dim == 3:
            ax.scatter(x[:, 0], x[:, 1], x[:, 2])
        elif dim == 2:
            ax.scatter(x[:, 0], x[:, 1])

        plt.show()
    else:
        print("Too many/too few dimensions to visualize")
        #https://matplotlib.org/stable/api/projections_api.html#module-matplotlib.projections

#setting up plot

#data collection
iris = datasets.load_iris()
x = iris.data
#actual decomposition
#len(x[0]) = upper bound

dimensions = 3
x = dataAnalysis(dimensions, x)
visualize(dimensions, x)
#plt.plot(x)
#plot is an additional plot (like the add subplot is unnecessary): reason for issue: was overriding previous plot
#however this is merely adding the terms
#need to display pca on plt

#math ee - perhaps describe how to do it with gradient function and the like to pick a random dimension
#instead of going down one dimension at a time
# : D D D D D 



#run pca
#setup for food
#contain in fubnc
#bin searhch
#change data set to word2vec


# print("Hi")

# import numpy as np
# import pandas as pd
# import mathplotlib.pyplot as plt
# from numpy import linalg as LA

# print("hi")

# #get semantle vector data set
# vectors = np.array([[1, 2], [7, 6], [4, 2]])
# print(vectors)

#for all dimensions until == 0
#simplify vector data set to that dimension (300, 299, etc.)
    #this is cause the procedure that would be coded by hand is down 1 dimension each time however
    #calling a pca function + bin searching for the criterion might work more efficiently
#run AI on each of them
#"judge" using criterion: 1. can AI get the word? 2. how many tries
#determine smallest dimension for vector set that has the AI getting the word in 3 or less tries