#Import pandas library for creating dataframe
import pandas as pd

#Import numpy for performing mathematical on the data
import numpy as np

#Import matplotlib for plotting interactive plots of data
import matplotlib.pyplot as plt

#Import standard scaler for Scaling data
from sklearn.preprocessing import StandardScaler

#Import PCA for Principal Component Analysis
from sklearn.decomposition import PCA

#Loading the data using pandas
data=pd.read_csv("thermodynamics.csv")

#Print the dataframe created using pandas
data

#Print first 5 rows of the data
data.head()

#Print last 5 rows of the data
data.tail()

#Print the information of the data type
data.info()

#Print the descriptive statistics of the data
data.describe()

#Print the dimensions of the data
data.shape

#Print the data type of every column of the data
data.dtypes

#Apply the Principal Component Anbalysis and set the number of components to 2
pca=PCA(n_components=2)
pca_data=pca.fit_transform(data)

#Print the reduced data
pca_data

#Convert the reduced data to the Data Frame
pca_df=pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])

#Print the Data Frame
pca_df

#Plot the scatter plot for the resultant Principal Components
plt.figure(figsize=(16, 12))
plt.scatter(pca_df.PCA1, pca_df.PCA2)
plt.xlabel('Principal Component 1', fontsize=24)
plt.ylabel('Principal Component 2', fontsize=24)
plt.title('Principal Component Analysis of the Thermodynamics data', fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()

#Plot the subplot (Color the principal components by PC1)
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['PCA1'], cmap='coolwarm', marker='o', edgecolor='k')
plt.title('PCA Scatter Plot Colored by PCA1')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(scatter, label='PCA1 Value')
plt.grid(True)

#Plot the subplot (Color the principal components by PC2)
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['PCA2'], cmap='coolwarm', marker='o', edgecolor='k')
plt.title('PCA Scatter Plot Colored by PCA2')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(scatter, label='PCA1 Value')
plt.grid(True)
