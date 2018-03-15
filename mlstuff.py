import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pprint import pprint

from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from pprint import pprint
# danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature

genres = {0:'jazz', 1:'rock', 2:'hip_hop'}
genre_x = [[] , [] , []]
genre_y = [[],[],[]]

genre_x_train = []
genre_y_train = []

genre_x_test = []
genre_y_test = []

def get_attributes():
    for key, value in genres.items():
        tibs = []

        with open('analysis_' + value + '_ids_train.txt', 'r') as f:
            for line in f:
                attributes = line.split(',')
                genre_x_train.append([float(x) for x in attributes[:len(attributes)-1]])
                genre_y_train.append(int(attributes[len(attributes)-1]))
            pprint (len(genre_x_train))

        with open('analysis_' + value + '_ids_train.txt', 'r') as f:
            for line in f:
                attributes = line.split(',')
                genre_x_test.append([float(x) for x in attributes[:len(attributes)-1]])
                genre_y_test.append(int(attributes[len(attributes)-1]))
            pprint (len(genre_x_train))


def plot_pca(X, y):
    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]
    iris = datasets.load_iris()
    # y = iris.target
    # print(len(y))
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = PCA(n_components=12)
    pca.fit(X)
    X = pca.transform(X)
    # pprint(X)
    # print(X[y == 0, 0].mean())
    for name, label in [('Jazz', 0), ('Rock', 1), ('Hip Hop', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def knn_plot(X, y):
    n_neighbors = math.sqrt(len(X))
    knn = KNeighborsClassifier()
    X_r2 = knn.fit(X, y)
    new_y = knn.predict(genre_x_test)
    compare_y(new_y, genre_y_test)

    h = 0.02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


    target_names = ['jazz','rock','hip_hop']
    colors = ['navy', 'turquoise', 'darkorange']
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))

    plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points

    plt.scatter(genre_x_test[:, 0], genre_x_test[:, 1], c=new_y, cmap=cmap_light,
                edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification using KNN")



def plot_lda(X, y):
    iris = datasets.load_iris()

    # X = iris.data
    # y = iris.target

    # print(type(X))
    target_names = ['jazz','rock','hip_hop']

    lda = LinearDiscriminantAnalysis(n_components=12)
    X_r2 = lda.fit(X, y).transform(X)
    new_y = lda.predict(genre_x_test)

    compare_y(new_y, genre_y_test)

    colors = ['navy', 'turquoise', 'darkorange']
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of music dataset')
    plt.show()


def logistic_regression(X, y):
    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    result = logreg.fit(X, y)
    new_y = logreg.predict(genre_x_test)
    compare_y(new_y,genre_y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot also the training points
    plt.scatter(genre_x_test[:, 0], genre_x_test[:, 1], c=new_y, cmap=cmap_light,
                edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification using Logistic Regression")
    plt.xticks(())
    plt.yticks(())

    plt.show()

def compare_y(new_y, old_y):
    count = 0
    errors = 0
    for index, y in enumerate(new_y):
        if y != old_y[index]:
            errors+=1
        count+=1
    print(errors/count)
    return (errors/count)

# plot_qda()
get_attributes()
genre_x_train = np.array(genre_x_train)
genre_y_train = np.array(genre_y_train)
genre_x_test = np.array(genre_x_test)
genre_y_test = np.array(genre_y_test)


# plot_pca(genre_x_train, genre_y_train)
scaler = preprocessing.MinMaxScaler()

genre_x_train = scaler.fit_transform(genre_x_train, genre_y_train)
genre_x_test = scaler.fit_transform(genre_x_test, genre_y_test)

# genre_x_test = preprocessing.scale(genre_x_test)
# genre_x_train = preprocessing.scale(genre_x_train)

plot_lda(genre_x_train, genre_y_train)
knn_plot(genre_x_train, genre_y_train)
logistic_regression(genre_x_train, genre_y_train)
