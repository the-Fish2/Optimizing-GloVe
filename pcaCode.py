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
        fig = plt.figure(num = 1, figsize = (4, 3))
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

dimensions = 2
x = dataAnalysis(dimensions, x)
visualize(dimensions, x)
#plt.plot(x)
#plot is an additional plot (like the add subplot is unnecessary): reason for issue: was overriding previous plot
#however this is merely adding the terms
#need to display pca on plt
