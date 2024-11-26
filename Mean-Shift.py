import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
file_path = r"C:\Users\Azooo\Downloads\Cont\Country-data.csv"
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna()


data = pd.get_dummies(data, drop_first=True)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA(n_components=2)  # Reduced to 2 dimensions
pca_data = pca.fit_transform(scaled_data)

#Estimate the bandwidth
bandwidth = estimate_bandwidth(pca_data, quantile=0.19, n_samples=500)

#Applying MeanShift
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(pca_data)

#Evaluate the clusters
labels = meanshift.labels_
if len(np.unique(labels)) > 1:
    silhouette_avg = silhouette_score(pca_data, labels)
    print(f'Silhouette Score: {silhouette_avg}')
else:
    print("Only one cluster found. Adjust the bandwidth parameter.")

#Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='viridis')
plt.title('MeanShift Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
