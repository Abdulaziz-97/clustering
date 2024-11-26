import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\Azooo\Downloads\Cont\Country-data.csv"
data = pd.read_csv(file_path)


data = data.dropna()
x=data.columns[1:]
X = data.columns[1:] 
y = data['country']
plt.figure(figsize=(15, 10))
for i, x in enumerate(X):
    plt.subplot(3, 3, i+1) 
    sns.histplot(data[x], kde=True)
    plt.title(x)
plt.tight_layout()
plt.show()

data = pd.get_dummies(data, drop_first=True)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
pca_data = pca.fit_transform(scaled_data)

#Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42,n_init=10)
kmeans.fit(pca_data)

#Evaluate the clusters
labels = kmeans.labels_
silhouette_avg = silhouette_score(pca_data, labels)
print(f'Silhouette Score: {silhouette_avg}')

#Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='viridis')
plt.title('K-means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
