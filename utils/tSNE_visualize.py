from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
def visualize_2D(X, user_ids, user_dicts, n_clusters, n_samples):
    #tsne = TSNE(n_components=2, random_state=0)
    tsne = PCA(n_components=2)
    #Project the data in 2D

    X_2d = tsne.fit_transform(X)
    #Visualize the data
    
    model_clusters = KMeans(n_clusters = n_clusters)
    Y = model_clusters.fit(X_2d)
    plt.figure(figsize=(6, 5))
    for label in range(n_clusters):
        plt.scatter(X_2d[Y == label, 0], X_2d[Y == label, 1], c=label, label=label)
    random_label = random.randint(0, n_clusters)
    plt.legend()
    plt.show()
    user_id_random_label = user_ids[Y == random_int]
    for i in range(n_samples):
        print(user_dicts[user_id_random_label[i]])
def visualize_3D(X, user_ids, user_dicts, n_clusters, n_samples):
    #tsne = TSNE(n_components=2, random_state=0)
    tsne = PCA(n_components=3)
    #Project the data in 2D

    X_3d = tsne.fit_transform(X)
    #Visualize the data
    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111, projection='3d')
    
    model_clusters = KMeans(n_clusters = n_clusters)
    Y = model_clusters.fit_predict(X_3d)
    plt.figure(figsize=(6, 5))
    for label in range(n_clusters):
        ax.scatter(X_3d[Y == label, 0], X_3d[Y == label, 1], X_3d[Y == label, 2], c=label, label=label)
    random_label = random.randint(0, n_clusters)
    plt.legend()
    plt.show()
    user_id_random_label = user_ids[Y == random_int]
    for i in range(n_samples):
        print(user_dicts[user_id_random_label[i]])

