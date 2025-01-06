import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.cluster import KMeans

ds = pd.read_csv('src/Mall_Customers.csv')

print(ds)

x = ds.iloc[:, [3, 4]].values

w = []

for i in range(1, 11):
    k = KMeans(n_clusters=i, init='k-means++', random_state=42)
    k.fit(x)
    w.append(k.inertia_)

mtp.plot(range(1, 11), w)
mtp.title('The Elbow Method Graph')
mtp.xlabel('Number of Clusters (k)')
mtp.ylabel('Inertia')
mtp.show()

k = KMeans(n_clusters=5, init='k-means++', random_state=42)
pred = k.fit_predict(x)

mtp.scatter(x[pred == 0, 0], x[pred == 0, 1], s=100, c='blue', label='Cluster 1')
mtp.scatter(x[pred == 1, 0], x[pred == 1, 1], s=100, c='green', label='Cluster 2')
mtp.scatter(x[pred == 2, 0], x[pred == 2, 1], s=100, c='red', label='Cluster 3')
mtp.scatter(x[pred == 3, 0], x[pred == 3, 1], s=100, c='cyan', label='Cluster 4')
mtp.scatter(x[pred == 4, 0], x[pred == 4, 1], s=100, c='magenta', label='Cluster 5')

mtp.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', marker='X')

mtp.title('Clusters of Mall Customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()

mtp.show()
