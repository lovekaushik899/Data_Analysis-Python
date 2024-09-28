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

#Create a copy of the data
x = data.copy()

#Initialize the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest

#Fitting of the model
isolation_forest.fit(x)

#Predict the anomalies
anomalies = isolation_forest.predict(x)

#Add predictions to the DataFrame
data['Anomaly'] = anomalies

#Visualization of the results
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Volume'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Volume using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Volume')
plt.axhline(y=data['Volume'].mean(), color='r', linestyle='--', label='Mean Volume')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Temperature'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Temperature using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.axhline(y=data['Temperature'].mean(), color='r', linestyle='--', label='Mean Temperature')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Pressure'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Pressure using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Pressure')
plt.axhline(y=data['Pressure'].mean(), color='r', linestyle='--', label='Mean Pressure')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Kinetic'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Kinetic using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Kinetic')
plt.axhline(y=data['Kinetic'].mean(), color='r', linestyle='--', label='Mean Kinetic')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Potential'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Potential using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Potential')
plt.axhline(y=data['Potential'].mean(), color='r', linestyle='--', label='Mean Potential')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Energy'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Energy using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Energy')
plt.axhline(y=data['Energy'].mean(), color='r', linestyle='--', label='Mean Energy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['Density'], c=data['Anomaly'], cmap='coolwarm', marker='o')
plt.title('Anomaly Detection in Density using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Density')
plt.axhline(y=data['Density'].mean(), color='r', linestyle='--', label='Mean Density')
plt.legend()
plt.show()
