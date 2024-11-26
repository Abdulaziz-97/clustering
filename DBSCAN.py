import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = r"C:\Users\Azooo\Downloads\Cont\Country-data.csv"
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Handle missing values (if any)
data = data.dropna()

# Convert non-numeric columns to numeric if necessary
# For example, using one-hot encoding for categorical variables
data = pd.get_dummies(data, drop_first=True)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_data = pca.fit_transform(scaled_data)

# Step 4: Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.7, min_samples=5)
dbscan.fit(pca_data)

# Step 5: Evaluate the clusters
labels = dbscan.labels_
if len(np.unique(labels)) > 1:
    silhouette_avg = silhouette_score(pca_data, labels)
    print(f'Silhouette Score: {silhouette_avg}')
else:
    print("Only one cluster found. Adjust the eps and min_samples parameters.")

# Step 6: Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='viridis')
plt.title('DBSCAN Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
