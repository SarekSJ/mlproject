import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from copy import copy
from pprint import pprint

from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from pprint import pprint
# danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature

genres = {0:'jazz', 1:'rock', 2:'hip_hop', 3:'classical'}
genre_x = [[] , [] , [], []]
genre_y = [[],[],[], []]

genre_x_train = []
genre_y_train = []

genre_x_test = []
genre_y_test = []

plot_yes = True

num_features = 12


def get_attributes():
    for key, value in genres.items():
        tibs = []

        with open('analysis_' + value + '_ids_train.txt', 'r') as f:
            for index,line in enumerate(f):
                attributes = line.split(',')
                # genre_x_train.append([attributes[0], attributes[1], attributes[6], attributes[7], ])

                genre_x_train.append([float(x) for x in attributes[:len(attributes)-1]])
                genre_y_train.append(int(attributes[len(attributes)-1]))

            # pprint (len(genre_x_train))

        with open('analysis_' + value + '_ids_test.txt', 'r') as f:
            for line in f:
                attributes = line.split(',')
                # genre_x_test.append([attributes[0], attributes[1], attributes[6], attributes[7]])
                genre_x_test.append([float(x) for x in attributes[:len(attributes)-1]])
                genre_y_test.append(int(attributes[len(attributes)-1]))
            # pprint (len(genre_x_train))
    print(genre_x_test)
def plot_errors(new_y, type):
    x_errors = []
    y_errors = []
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA'])
    if plot_yes:

        for index, y in enumerate(new_y):
            if y != genre_y_test[index]:
                x_errors.append(genre_x_test[index])
                y_errors.append(y)
        x_errors = np.array(x_errors)
        y_errors = np.array(y_errors)
        plt.scatter(x_errors[:, 0], x_errors[:, 1], c=y_errors, cmap=cmap_light,
                    edgecolor='k', s=20)
        plt.xlabel("Dancebility")
        plt.ylabel("Energy")
        plt.title("Errors for : " + type)
        plt.show()

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
    for name, label in [('Jazz', 0), ('Rock', 1), ('Hip Hop', 2), ('Classical', 3)]:
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
    print('Knn: ')

    new_y = knn.predict(X)
    print('Train Error: ' + str(compare_y(new_y, y)))


    new_y = knn.predict(genre_x_test)
    print('Test Error: ' + str(compare_y(new_y, genre_y_test)))
    # plot_errors(new_y, "Knn")

    h = 0.02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FAFAAA'])
    cmap_light = ListedColormap(['#fc8d59', '#ffffbf', '#91bfdb', '#bc1111'])
    cmap_light = ListedColormap(['navy', 'turquoise', 'darkorange', 'red'])

    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


    target_names = ['jazz','rock','hip_hop', 'classical']
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])


    plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points

    plt.scatter(genre_x_test[:, 0], genre_x_test[:, 1], c=new_y, cmap=cmap_light,
                s=0.5, alpha = .8)
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    plt.xlabel("Dancebility")
    plt.ylabel("Energy")
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.title("KNN for the Music Dataset")
    plt.show()



def plot_lda(X, y):
    iris = datasets.load_iris()

    # X = iris.data
    # y = iris.target

    # print(type(X))

    print('LDA: ')

    target_names = ['jazz','rock','hip_hop', 'classical']

    lda = LinearDiscriminantAnalysis(n_components=num_features)
    X_r2 = lda.fit(X, y).transform(X)

    new_y = lda.predict(X)
    print("Train error rate: " + str(compare_y(new_y, y)))

    new_y = lda.predict(genre_x_test)
    print("Test error rate: " + str(compare_y(new_y, genre_y_test)))
    plot_errors(new_y, "LDA")

    colors = ['navy', 'turquoise', 'darkorange', 'red']
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name, s=0.5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of music dataset')
    plt.xlabel("Dancebility")
    plt.ylabel("Energy")
    plt.show()

def plot_qda(X, y):
    print('QDA: ')
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X,y)

    new_y = qda.predict(X)
    print("Train error rate: " + str(compare_y(new_y, y)))

    new_y = qda.predict(genre_x_test)
    print("Test error rate: " + str(compare_y(new_y, genre_y_test)))

    plot_errors(new_y, "QDA")

    colors = ['navy', 'turquoise', 'darkorange', 'red']
    target_names = ['jazz', 'rock', 'hip_hop', 'classical']

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.8, color=color,
                    label=target_name, s=.5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('QDA of music dataset')
    plt.xlabel("Dancebility")
    plt.ylabel("Energy")
    plt.show()


def logistic_regression(X, y):
    print('Logistic Regression: ')
    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    result = logreg.fit(X, y)

    new_y = logreg.predict(X)
    error_rate = compare_y(new_y,y)
    print("Train error rate: " + str(error_rate))


    new_y = logreg.predict(genre_x_test)
    error_rate = compare_y(new_y,genre_y_test)
    print("Test error rate: " + str(error_rate))
    # plot_errors(new_y, "Logistic Regression")


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FAFAAA'])
    cmap_light = ListedColormap(['navy', 'turquoise', 'darkorange', 'red'])

    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot also the training points
    plt.scatter(genre_x_test[:, 0], genre_x_test[:, 1], c=new_y, cmap=cmap_light,
                 s=.5, alpha=.8)
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    plt.xlabel("Dancebility")
    plt.ylabel("Energy")
    # # plt.xlim(xx.min(), xx.max())
    # # plt.ylim(yy.min(), yy.max())
    plt.title("Logistic Regression for the Music Dataset")
    # plt.xticks(())
    # plt.yticks(())
    #
    plt.show()

def compare_y(new_y, old_y):
    count = 0
    errors = 0
    for index, y in enumerate(new_y):
        if y != old_y[index]:
            errors+=1
        count+=1
    # print(errors/count)
    return (errors/count)

# plot_qda()
get_attributes()
genre_x_train = np.array(genre_x_train)
genre_y_train = np.array(genre_y_train)
genre_x_test = np.array(genre_x_test)
genre_y_test = np.array(genre_y_test)


genre_x_test = preprocessing.scale(genre_x_test)
genre_x_train = preprocessing.scale(genre_x_train)
# genre_x_test = genre_x_train
# plot_pca(genre_x_train, genre_y_train)
# plot_lda(genre_x_train, genre_y_train)
# plot_qda(genre_x_train, genre_y_train)
#
# knn_plot(copy(genre_x_train), genre_y_train)
logistic_regression(genre_x_train, genre_y_train)
