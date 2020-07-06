from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
import sys

filename = sys.argv[1]
k = int(sys.argv[2])
#import the dataset
df = pd.read_csv(filename, delimiter=" ")

x = df.drop(columns=['Class'])

kmeans = KMeans(n_clusters=k, precompute_distances="auto", n_jobs=-1)
df['Clusters'] = kmeans.fit_predict(x)
value = df.values.tolist()
print(Counter(kmeans.labels_))
#print(df)

