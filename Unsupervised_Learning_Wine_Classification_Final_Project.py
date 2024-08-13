#!/usr/bin/env python
# coding: utf-8

# # **Unsupervised Learning for Wine Classification - Final Project**
# 
# ### **A Clustering Approach to Identify Natural Groupings Based on Chemical Properties**

# Git: https://github.com/AnushaK7018/UnsupervisedLearningFinal

# # **Project Objective:**
# 
# The primary objective of this project was to explore and apply unsupervised machine learning techniques, specifically clustering algorithms, to identify natural groupings within a dataset. The dataset, consisting of various chemical properties of wines, was used to segment the wine samples into distinct clusters based on their similarities. The ultimate aim was to determine which clustering method best captured the underlying structure of the data, providing meaningful and actionable insights.

# ### **Data Description:**
# 
# **Dataset Name:** Wine Dataset
# 
# **Source:** https://archive.ics.uci.edu/dataset/109/wine (UCI Machine Learning Repository)
# 
# **Number of Samples:** 178 (after removing outliers)
# 
# **Features:** The dataset includes 13 features that represent different chemical properties of the wine, each of which is a continuous variable. These features were standardized to ensure that they have a mean of zero and a standard deviation of one, which is essential for clustering algorithms to perform optimally.
# 
# Alcohol: Alcohol content in the wine.
# 
# Malic Acid: Malic acid concentration.
# 
# Ash: Ash content.
# 
# Alcalinity of Ash: Measure of the alkalinity of ash in the wine.
# 
# Magnesium: Magnesium concentration.
# 
# Total Phenols: Total phenol content.
# 
# Flavanoids: Flavonoid content, an indicator of antioxidant capacity.
# 
# Nonflavanoid Phenols: Phenolic compounds that are not flavonoids.
# 
# Proanthocyanins: Tannins found in wine, contributing to astringency.
# 
# Color Intensity: Intensity of the wine’s color.
# 
# Hue: The hue of the wine.
# 
# OD280/OD315 of Diluted Wines: Ratio of absorbance at 280nm and 315nm,
# indicative of the wine's phenolic content.
# 
# Proline: An amino acid found in wine, often used as a marker for identifying wine types.
# 
# Target Variable: Not used in the clustering analysis, but represents the wine cultivar (used only for validation or supervised learning tasks).

# ## **Exploratory Data Analysis (EDA)**
# 
# Size: The dataset contains 178 samples (after removing outliers).
# 
# Missing Values: The dataset was inspected and confirmed to have no missing values, eliminating the need for data imputation.
# 
# Outliers: Outliers were identified and removed using the Interquartile Range (IQR) method, leading to a cleaner dataset with more representative clusters.
# 
# Skewness: Skewness was checked for each feature, and transformations were applied where necessary to reduce the skewness, further preparing the data for clustering.

# ### **Unsupervised Learning Problem:**
# 
# Given the nature of the dataset, the primary unsupervised learning problem addressed in this project is clustering. The objective was to group the wine samples into clusters based on their chemical properties without using the target labels (cultivars).
# 
# **Clustering Techniques Applied:**
# 
# **K-Means Clustering:** Chosen for its simplicity and effectiveness in handling spherical clusters. K-Means was expected to provide clear, well-defined clusters due to the nature of the data.
# 
# **Gaussian Mixture Models (GMM):** Applied to allow for more flexibility in cluster shapes, particularly elliptical clusters. GMM was selected to see if it could provide a better fit for the data's inherent structure.
# 
# **Hierarchical Clustering (Ward Linkage):** Used to explore the hierarchical relationships between clusters. This method was particularly useful for understanding the dendrogram structure and identifying optimal clusters without specifying the number of clusters upfront.
# 
# **DBSCAN:** Although initially considered for its robustness to noise and ability to identify arbitrarily shaped clusters, DBSCAN did not perform well on this dataset, likely due to the chosen parameter settings.
# 
# ***Resulting Cluster Analysis:***
# 
# **Optimal Clustering Method:** Through a comparative analysis of silhouette scores and visual inspections, K-Means was determined to be the most effective clustering method, yielding the highest silhouette score and producing well-separated, meaningful clusters.
# 
# **Cluster Interpretability:** Each method's clusters were analyzed, and K-Means provided the most interpretable and actionable segments, making it the preferred method for this dataset.

# In[ ]:


import pandas as pd
from google.colab import drive
drive.mount('/content/drive')


# # **Import Data**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Load the dataset
column_names = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# Load the data
data = pd.read_csv('/content/drive/My Drive/wine/wine.data', header=None, names=column_names)
data.head(5)


# In[ ]:


# Describe the dataset
data_description = data.describe()
print(data_description)


# # **Exploratory Data Analysis (EDA) - Inspect, Visualize, and Clean the Data**

# In[ ]:


# Plot histograms for each feature
plt.figure(figsize=(15, 10))
data.hist(bins=15, figsize=(15, 10))
plt.show()


# In[ ]:



# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:



# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Silhouette score
silhouette_avg = silhouette_score(scaled_data, kmeans_labels)
print(silhouette_avg)


# In[ ]:


# Perform hierarchical clustering
Z = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=12)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()


# In[ ]:



# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot PCA
plt.figure(figsize=(10, 7))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=100)
plt.title('PCA - Wine Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[ ]:


# Check for missing values
missing_values = data.isnull().sum()

# Box plots for each feature
plt.figure(figsize=(15, 10))
data.boxplot()
plt.title('Boxplot of All Features')
plt.xticks(rotation=45)
plt.show()


# # **Remove the outliers and reanalyze.**

# In[ ]:


# Identify outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

# Check for skewness in data
skewness = data.skew()

# Log transformation on skewed data
log_transformed_data = data.copy()
for column in data.columns:
    if skewness[column] > 0.5 or skewness[column] < -0.5:
        log_transformed_data[column] = np.log1p(data[column])

# Skewness after log transformation
log_skewness = log_transformed_data.skew()


missing_values, outliers, skewness, log_skewness


# In[ ]:




# Visualize the effect of log transformation on skewed features
plt.figure(figsize=(15, 10))
log_transformed_data.hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Features After Log Transformation')
plt.show()


# In[ ]:


# Removing outliers using the IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Define a condition for outliers
condition = ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)

# Remove outliers
data_no_outliers = data[condition]

# Re-check for outliers after removal
outliers_after_removal = ((data_no_outliers < (Q1 - 1.5 * IQR)) | (data_no_outliers > (Q3 + 1.5 * IQR))).sum()

# Describe the dataset after outlier removal
data_no_outliers_description = data_no_outliers.describe()

# Re-check skewness after outlier removal
skewness_after_removal = data_no_outliers.skew()


outliers_after_removal, data_no_outliers_description, skewness_after_removal


# In[ ]:


# Visualize the data distribution after outlier removal
plt.figure(figsize=(15, 10))
data_no_outliers.hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Features After Outlier Removal')
plt.show()


# ### **Analysis**
# 
# **1. Missing Values:**
# 
# There are no missing values in the dataset, which means no imputation or data removal is needed for missing data.
# 
# **2. Outliers:**
# 
# The dataset contains a few outliers in the following features:
# 
# Malic acid: 3 outliers
# 
# Ash: 3 outliers
# 
# Alcalinity of ash: 4 outliers
# 
# Magnesium: 4 outliers
# 
# Proanthocyanins: 2 outliers
# 
# Color intensity: 4 outliers
# 
# Hue: 1 outlier
# 
# These outliers might be significant, especially in features like Malic acid and Magnesium, which are chemical properties that could vary naturally but need further analysis depending on the model to be used.
# 
# **3. Skewness:**
# 
# Several features exhibit skewness:
# 
# Malic acid: Skewness = 1.04 (positive skew)
# 
# Magnesium: Skewness = 1.10 (positive skew)
# 
# Proanthocyanins: Skewness = 0.52 (positive skew)
# 
# Color intensity: Skewness = 0.87 (positive skew)
# 
# Proline: Skewness = 0.77 (positive skew)
# 
# Skewness in data might affect model performance, especially in algorithms sensitive to data distribution like linear models.
# 
# **4. Log Transformation:**
# 
# After applying log transformation to skewed features, the skewness of these features is significantly reduced, making them closer to a normal distribution:
# 
# Malic acid: Reduced to 0.53
# 
# Magnesium: Reduced to 0.61
# 
# Proanthocyanins: Reduced to -0.17
# 
# Color intensity: Reduced to 0.10
# 
# Proline: Reduced to 0.09
# 
# The log transformation has successfully reduced the skewness in most of the skewed features, which might be beneficial for the analysis.

# In[ ]:


# Reapply KMeans clustering on the data without outliers
kmeans_no_outliers = KMeans(n_clusters=3, random_state=42)
kmeans_labels_no_outliers = kmeans_no_outliers.fit_predict(scaled_data[condition])

# Silhouette score for KMeans without outliers
silhouette_avg_no_outliers = silhouette_score(scaled_data[condition], kmeans_labels_no_outliers)

# Reapply hierarchical clustering on the data without outliers
Z_no_outliers = linkage(scaled_data[condition], method='ward')

silhouette_avg_no_outliers


# In[ ]:



# Plot dendrogram for data without outliers
plt.figure(figsize=(10, 7))
dendrogram(Z_no_outliers, truncate_mode='lastp', p=12)
plt.title('Dendrogram for Hierarchical Clustering (No Outliers)')
plt.show()


# In[ ]:


# Reapply PCA on the data without outliers
pca_no_outliers = PCA(n_components=2)
pca_data_no_outliers = pca_no_outliers.fit_transform(scaled_data[condition])

# Plot PCA for data without outliers
plt.figure(figsize=(10, 7))
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=kmeans_labels_no_outliers, cmap='viridis', edgecolor='k', s=100)
plt.title('PCA - Wine Data (No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# ### **Post-Outlier Removal Analysis**
# 
# After removing the outliers, here's the updated analysis:
# 
# **1. Outliers Check:**
# 
# All outliers have been successfully removed, as indicated by zero counts of outliers in all features.
# 
# **2. Data Description:**
# 
# The dataset now contains 161 samples, down from the original 178.
# 
# The descriptive statistics (mean, standard deviation, etc.) have slightly changed due to the removal of outliers, providing a more robust view of the central tendencies and variances.
# 
# **3. Skewness:**
# 
# The skewness of several features has been reduced:
# 
# Malic acid: Reduced to 0.89 (still slightly skewed).
# 
# Magnesium: Reduced to 0.50 (moderately skewed).
# 
# Proline: Reduced to 0.70 (moderately skewed).
# 
# Other features like Nonflavanoid phenols and Color intensity still exhibit some skewness, though they are less pronounced.
# 
# **4. Data Distribution:**
# 
# The histograms show the updated distribution of the data after removing outliers, with distributions appearing more centralized and less spread.
# 
# Removing outliers has had a positive impact on the clustering results, as evidenced by the improved silhouette score and clearer separation in the PCA plot. The data now forms more distinct clusters, which could lead to better insights when analyzing the natural groupings within the wine dataset

# # **Compare K-Means and Hierarchical clustering using plot**

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

# K-Means clustering
kmeans_labels_no_outliers = kmeans_no_outliers.labels_

# Hierarchical clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=3)
hierarchical_labels_no_outliers = hierarchical_clustering.fit_predict(scaled_data[condition])

# Plot K-Means vs Hierarchical clustering using PCA components
plt.figure(figsize=(15, 6))

# K-Means
plt.subplot(1, 2, 1)
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=kmeans_labels_no_outliers, cmap='viridis', edgecolor='k', s=100)
plt.title('K-Means Clustering (No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Hierarchical
plt.subplot(1, 2, 2)
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=hierarchical_labels_no_outliers, cmap='viridis', edgecolor='k', s=100)
plt.title('Hierarchical Clustering (No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()


# ### **K-Means Clustering:**
# 
# The left plot shows the clusters formed by K-Means. The clusters are more uniform in size and shape, which is a characteristic of the K-Means algorithm. The algorithm tries to minimize the variance within each cluster, leading to spherical clusters.
# 
# ### **Hierarchical Clustering:**
# 
# The right plot shows the clusters formed by Hierarchical Clustering. This method doesn't assume a fixed shape for clusters, leading to potentially more irregular cluster shapes. Hierarchical clustering can capture more complex relationships between data points, which is evident in the slightly different cluster formation.

# # **K-Means Hyperparameter Tuning:**

# In[ ]:


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Range of clusters to try
k_values = range(2, 11)

# Store inertia and silhouette scores
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data[condition])
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data[condition], kmeans.labels_))

# Plotting Inertia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

# Plotting Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Optimal k based on silhouette score
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
optimal_k, max(silhouette_scores)


# # **K-Means Hyperparameter Tuning Results**
# 
# **Elbow Method:** The plot on the left shows the inertia values for different numbers of clusters (k). The "elbow" point, where the inertia starts to decrease more slowly, can suggest a reasonable choice for the number of clusters.
# 
# **Silhouette Score:** The plot on the right shows the silhouette score for different values of k. The silhouette score peaks at
# 𝑘
# =
# 3
# k=3, indicating that this might be the optimal number of clusters.
# 
# **Optimal k:**
# 
# Based on the silhouette score, the optimal number of clusters is 3, with a silhouette score of approximately 0.309.

# **Optimize Hierarchical Clustering parameters next.**

# In[ ]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# Linkage methods to try
linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters_range = range(2, 11)

# Store silhouette scores
silhouette_scores_hc = {}

for method in linkage_methods:
    method_scores = []
    for n_clusters in n_clusters_range:
        hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = hierarchical_clustering.fit_predict(scaled_data[condition])
        silhouette_avg = silhouette_score(scaled_data[condition], labels)
        method_scores.append(silhouette_avg)
    silhouette_scores_hc[method] = method_scores

# Plotting Silhouette Scores for different linkage methods
plt.figure(figsize=(12, 8))
for method in linkage_methods:
    plt.plot(n_clusters_range, silhouette_scores_hc[method], marker='o', label=f'{method} linkage')

plt.title('Silhouette Scores for Hierarchical Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()

# Finding the best combination
best_method = max(silhouette_scores_hc, key=lambda x: max(silhouette_scores_hc[x]))
best_n_clusters = n_clusters_range[silhouette_scores_hc[best_method].index(max(silhouette_scores_hc[best_method]))]
best_silhouette_score = max(silhouette_scores_hc[best_method])

best_method, best_n_clusters, best_silhouette_score


# **Hierarchical Clustering Hyperparameter Tuning Results**
# 
# **Silhouette Scores for Different Linkage Methods:** The plot shows the silhouette scores for different numbers of clusters (ranging from 2 to 10) across four linkage methods: ward, complete, average, and single.
# Optimal Parameters:
# 
# **Best Linkage Method:** The ward linkage method provides the best performance.
# Optimal Number of Clusters: The optimal number of clusters is 3, with a silhouette score of approximately 0.302.
# 
# The ward linkage with 3 clusters appears to be the most effective configuration for hierarchical clustering based on the silhouette score.
# This is in line with the results from K-Means, which also suggested 3 clusters as the optimal number.

# # **Compare K-Means and Hierarchical Results**

# In[ ]:


# Plot K-Means vs Hierarchical clustering using PCA components
plt.figure(figsize=(15, 6))

# K-Means
plt.subplot(1, 2, 1)
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=kmeans_labels_no_outliers, cmap='viridis', edgecolor='k', s=100)
plt.title('K-Means Clustering (No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Hierarchical
plt.subplot(1, 2, 2)
hierarchical_clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_method)
hierarchical_labels_no_outliers = hierarchical_clustering.fit_predict(scaled_data[condition])
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=hierarchical_labels_no_outliers, cmap='viridis', edgecolor='k', s=100)
plt.title('Hierarchical Clustering (Ward Linkage, No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()


# The plots above visually compare the clustering results from K-Means (left) and Hierarchical Clustering using the Ward linkage method (right) on the PCA-reduced data:
# 
# **Cluster Separation:**
# 
# ***K-Means:*** The clusters appear more evenly sized and well-separated, which is expected as K-Means optimizes for spherical cluster shapes. The algorithm tries to minimize the variance within each cluster, leading to more regular, circular clusters.
# 
# ***Hierarchical Clustering (Ward):*** The clusters formed by hierarchical clustering are slightly more irregular in shape, which suggests that this method can capture more complex cluster structures. The clusters are also well-separated, though the shapes are not as uniform as those in K-Means.
# 
# **Silhouette Scores:**
# 
# K-Means achieved a slightly higher silhouette score (0.309) compared to hierarchical clustering (0.302), indicating that K-Means may have slightly better-defined clusters in this case.
# 
# **Cluster Structure:**
# 
# K-Means may be more suitable if the true underlying clusters are close to spherical and of similar size.
# 
# Hierarchical Clustering is advantageous if you suspect the clusters might have more irregular shapes or if you want to explore the hierarchical structure of the data.

# **Analysis:**
# 
# ***K-Means:*** Slightly better in terms of silhouette score, and works well when clusters are spherical and evenly sized.
# 
# ***Hierarchical Clustering:*** More flexible in capturing complex shapes, but with a marginally lower silhouette score.

# # **Compare Other Clustering Methods:**

# Compare other clustering methods to see how they perform on this dataset.
# 
# **1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
# DBSCAN is a density-based clustering method that identifies clusters as dense regions in the data. It's particularly useful for identifying clusters of varying shapes and handling outliers.
# 
# **2. Gaussian Mixture Models (GMM):**
# GMM is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions. It can model clusters that are more elliptical and is more flexible than K-Means.
# 
# **3. Agglomerative Clustering with Different Linkages:**
# While we have used the Ward linkage, we can also try other linkages like complete, average, or single linkage to see if they provide better results.

# **Implement DBSCAN and GMM for comparison**

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data[condition])

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(scaled_data[condition])

# Silhouette scores
dbscan_silhouette = silhouette_score(scaled_data[condition], dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
gmm_silhouette = silhouette_score(scaled_data[condition], gmm_labels)

# Plot DBSCAN vs GMM using PCA components
plt.figure(figsize=(15, 6))

# DBSCAN
plt.subplot(1, 2, 1)
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=dbscan_labels, cmap='viridis', edgecolor='k', s=100)
plt.title(f'DBSCAN Clustering (No Outliers)\nSilhouette Score: {dbscan_silhouette:.3f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# GMM
plt.subplot(1, 2, 2)
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=gmm_labels, cmap='viridis', edgecolor='k', s=100)
plt.title(f'GMM Clustering (No Outliers)\nSilhouette Score: {gmm_silhouette:.3f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

dbscan_silhouette, gmm_silhouette


# The plots above visualize the clustering results from DBSCAN and GMM using the PCA-reduced 2D data:
# 
# **DBSCAN:**
# 
# ***Silhouette Score:*** The silhouette score for DBSCAN is -1, which indicates that the clustering might not have performed well. This could be due to DBSCAN identifying most points as noise (labeled as -1), leading to poorly defined clusters.
# 
# ***Cluster Visualization:*** The PCA plot shows that DBSCAN might not have identified clear clusters in this dataset, likely due to the parameter choices (eps and min_samples). DBSCAN is more sensitive to these parameters and might need fine-tuning.
# 
# **Gaussian Mixture Model (GMM):**
# 
# ***Silhouette Score:*** The silhouette score for GMM is 0.306, which is close to the scores obtained by K-Means and Hierarchical Clustering.
# 
# ***Cluster Visualization:*** The PCA plot for GMM shows more elliptical clusters compared to the spherical ones typically produced by K-Means. GMM can better model clusters with varying shapes and covariance structures.

# **Summary of Clustering Method Comparisons**
# 
# ***K-Means:*** Best silhouette score (0.309), works well with spherical clusters.
# 
# ***Hierarchical Clustering (Ward):*** Comparable performance to K-Means with a silhouette score of 0.302, captures more complex relationships.
# 
# ***GMM:*** Similar silhouette score to K-Means (0.306), allows for more flexible, elliptical clusters.
# 
# ***DBSCAN:*** Did not perform well with the current parameter settings (silhouette score -1), but could be improved with parameter tuning.

# # **Tune DBSCAN Parameters for Better Results**

# In[ ]:


from sklearn.model_selection import ParameterGrid

# Define the parameter grid for DBSCAN
param_grid = {
    'eps': np.arange(0.2, 1.0, 0.1),
    'min_samples': range(3, 10)
}

# Initialize variables to store the best parameters and score
best_params = None
best_silhouette = -1

# Perform grid search
for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(scaled_data[condition])
    # Check if more than 1 cluster is formed
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(scaled_data[condition], labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_params = params

best_params, best_silhouette


# It appears that the grid search for DBSCAN did not yield any improvement, with the silhouette score remaining at -1. This suggests that DBSCAN, with the current range of parameters, is not identifying meaningful clusters in this dataset.

# # **Agglomerative Clustering with Different Linkages**

# In[ ]:


# Linkage methods to try for Agglomerative Clustering
linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters = 3  # As determined optimal for this dataset

# Store silhouette scores for each linkage method
silhouette_scores_linkages = {}

for method in linkage_methods:
    agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = agglomerative_clustering.fit_predict(scaled_data[condition])
    silhouette_avg = silhouette_score(scaled_data[condition], labels)
    silhouette_scores_linkages[method] = silhouette_avg

silhouette_scores_linkages


# **Observations:**
# 
# Ward Linkage continues to provide the best performance with a silhouette score of 0.302.
# 
# Complete Linkage follows with a silhouette score of 0.268.
# 
# Average Linkage performs slightly worse, with a silhouette score of 0.236.
# 
# Single Linkage has the lowest silhouette score (0.070), indicating that it may not be the best choice for this dataset.
# 
# **Analysis:**
# 
# Ward Linkage remains the most effective method for Agglomerative Clustering on this dataset.
# 
# Complete Linkage could be an alternative if we want to explore different cluster formations, though it performs slightly worse than Ward.
# 
# Single Linkage does not perform well in this context, likely due to its tendency to create long, stringy clusters.

# # **Visualize the Best Clustering Method**

# In[ ]:


# Visualize the best clustering method: Agglomerative Clustering with Ward linkage
best_agglomerative_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
best_agglomerative_labels = best_agglomerative_clustering.fit_predict(scaled_data[condition])

# Plot the clusters using PCA components
plt.figure(figsize=(10, 7))
plt.scatter(pca_data_no_outliers[:, 0], pca_data_no_outliers[:, 1], c=best_agglomerative_labels, cmap='viridis', edgecolor='k', s=100)
plt.title('Agglomerative Clustering (Ward Linkage, No Outliers)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# The plot above shows the clusters formed by Agglomerative Clustering with Ward linkage using the first two principal components of the PCA. This method was identified as the best clustering approach based on the silhouette score.

# ***detailed cluster statistics for each method***

# In[ ]:


import pandas as pd
from IPython.display import display
# Prepare a function to calculate cluster statistics
def cluster_statistics(data, labels):
    cluster_stats = pd.DataFrame()
    cluster_stats['Cluster'] = np.unique(labels)
    cluster_stats['Size'] = [np.sum(labels == i) for i in np.unique(labels)]
    cluster_stats['Mean'] = [np.mean(data[labels == i], axis=0) for i in np.unique(labels)]
    cluster_stats['StdDev'] = [np.std(data[labels == i], axis=0) for i in np.unique(labels)]
    return cluster_stats


# In[ ]:


# K-Means statistics
kmeans_stats = cluster_statistics(scaled_data[condition], kmeans_labels_no_outliers)

display(kmeans_stats)


# In[ ]:



# Hierarchical (Ward) statistics
hierarchical_stats = cluster_statistics(scaled_data[condition], hierarchical_labels_no_outliers)
display(hierarchical_stats)


# In[ ]:


# GMM statistics
gmm_stats = cluster_statistics(scaled_data[condition], gmm_labels)
display(gmm_stats)


# In[ ]:


# Agglomerative Clustering (Ward) statistics
agglomerative_stats = cluster_statistics(scaled_data[condition], best_agglomerative_labels)
display(agglomerative_stats)


# In[ ]:


# Compare silhouette scores across the different methods
silhouette_scores_comparison = {
    "K-Means": max(silhouette_scores),
    "Hierarchical (Ward)": silhouette_scores_hc['ward'],
    "Hierarchical (Complete)": silhouette_scores_hc['complete'],
    "Hierarchical (Average)": silhouette_scores_hc['average'],
    "Hierarchical (Single)": silhouette_scores_hc['single'],
    "GMM": gmm_silhouette,
    "DBSCAN": dbscan_silhouette
}

silhouette_scores_comparison


# **Summary:**
# 
# 
# 
# *   K-Means and GMM have the highest silhouette scores, with K-Means slightly
# outperforming GMM.
# 
# *   Hierarchical Clustering with Ward Linkage is close behind, with a silhouette score of 0.302.
# 
# *   Complete and Average Linkage methods perform moderately well, but not as well as Ward Linkage.
# 
# *   Single Linkage performs poorly in this context, likely due to its tendency to create long, stringy clusters.
# 
# *   DBSCAN did not perform well with the current parameter settings.
# 
# Based on these scores, K-Means and GMM are the top-performing clustering methods for this dataset, with Hierarchical Clustering (Ward Linkage) being a strong alternative.

# In[ ]:


# Extracting the means for visualization
kmeans_means = np.array(kmeans_stats['Mean'].tolist())
hierarchical_means = np.array(hierarchical_stats['Mean'].tolist())
gmm_means = np.array(gmm_stats['Mean'].tolist())
agglomerative_means = np.array(agglomerative_stats['Mean'].tolist())

# Plotting cluster means for comparison
plt.figure(figsize=(15, 10))

# K-Means
plt.subplot(2, 2, 1)
plt.plot(kmeans_means.T, marker='o')
plt.title('K-Means Cluster Means')
plt.xlabel('Features')
plt.ylabel('Mean Values')

# Hierarchical (Ward)
plt.subplot(2, 2, 2)
plt.plot(hierarchical_means.T, marker='o')
plt.title('Hierarchical Clustering (Ward) Cluster Means')
plt.xlabel('Features')
plt.ylabel('Mean Values')

# GMM
plt.subplot(2, 2, 3)
plt.plot(gmm_means.T, marker='o')
plt.title('GMM Cluster Means')
plt.xlabel('Features')
plt.ylabel('Mean Values')

# Agglomerative (Ward)
plt.subplot(2, 2, 4)
plt.plot(agglomerative_means.T, marker='o')
plt.title('Agglomerative Clustering (Ward) Cluster Means')
plt.xlabel('Features')
plt.ylabel('Mean Values')

plt.tight_layout()
plt.show()


# All methods exhibit distinct cluster means, indicating that each clustering approach identifies different groupings in the data.
# K-Means and Ward linkage in hierarchical clustering tend to produce more similar cluster profiles, while GMM introduces some variations, likely due to its more flexible modeling approach

# # **Summary of Results:**
# 
# ### **K-Means:**
# 
# **Silhouette Score:** 0.309 (highest among the methods tried).
# 
# **Performance:** Consistently produced well-separated, spherical clusters.
# 
# **Interpretability:** Easy to interpret and implement.
# 
# ### **Gaussian Mixture Model (GMM):**
# 
# **Silhouette Score:** 0.306 (very close to K-Means).
# 
# **Performance**: Capable of modeling elliptical clusters, providing a probabilistic assignment of points to clusters.
# 
# **Flexibility:** More flexible in handling clusters of different shapes.
# 
# ### **Hierarchical Clustering (Ward Linkage):**
# 
# **Silhouette Score:** 0.302 (slightly lower but still competitive).
# 
# **Performance:** Produces clusters similar to K-Means, with a strong hierarchical structure.
# 
# **Interpretability:** Offers insights into cluster relationships through dendrograms.
# 
# ### **DBSCAN:**
# 
# **Silhouette Score:** -1 (poor performance with current parameter settings).
# 
# **Performance:** Failed to produce meaningful clusters with the given data and parameters.

# ### **Best Method for This Project: K-Means**
# 
# **Highest Silhouette Score:** K-Means achieved the highest silhouette score, indicating that it created well-defined and separated clusters in your dataset.
# 
# **Simplicity and Interpretability:** K-Means is straightforward to implement and interpret. It’s easy to understand which data points belong to which clusters and why.
# 
# **Consistency Across Clusters:** The K-Means algorithm produced consistent, evenly sized clusters, which aligns well with datasets where clusters are expected to be spherical and roughly equal in size.
# 
# **Speed and Scalability:** K-Means is computationally efficient, making it a good choice for larger datasets or when you need to quickly obtain clustering results.
# 
# **Close Performance to GMM:** While GMM provides more flexibility in terms of cluster shape, the marginal difference in silhouette score (0.309 vs. 0.306) suggests that K-Means' simpler approach is sufficient and effective for this dataset.

# # **Conclusion:**
# 
# This project successfully demonstrated the application of unsupervised learning techniques to identify meaningful clusters within the dataset. K-Means was determined to be the optimal method due to its superior performance, simplicity, and interpretability. The clustering results provide a solid foundation for decision-making in various business contexts, from customer segmentation to operational efficiency.
# 
# **Key Takeaways:**
# 
# Further Analysis: Further fine-tuning of the clustering methods, particularly DBSCAN, could be explored to see if alternative parameter settings yield better results.
# 
# **Final Thoughts:**
# 
# This project highlights the importance of careful data preparation, thorough exploration of clustering methods, and thoughtful interpretation of results in deriving actionable insights from unsupervised learning.
