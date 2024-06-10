import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Crear el dataset
data = {
    'Punto Origen': ['A', 'A', 'B', 'B', 'C'],
    'Punto Destino': ['B', 'C', 'C', 'D', 'D'],
    'Distancia': [4, 2, 5, 10, 3]
}
df = pd.DataFrame(data)
print("Dataset inicial:")
print(df)

# Codificar las variables categóricas
label_encoder = LabelEncoder()
df['Punto Origen'] = label_encoder.fit_transform(df['Punto Origen'])
df['Punto Destino'] = label_encoder.fit_transform(df['Punto Destino'])
print("\nDataset después de la codificación:")
print(df)

# Preparar los datos para el clustering
X = df[['Punto Origen', 'Punto Destino', 'Distancia']]

# Crear el modelo de K-means con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Añadir la información del cluster al DataFrame original
df['Cluster'] = kmeans.labels_
print("\nDataset con clusters asignados:")
print(df)

# Visualizar los clusters
plt.scatter(df['Punto Origen'], df['Punto Destino'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Punto Origen')
plt.ylabel('Punto Destino')
plt.title('Clusters de Puntos')
plt.show()
