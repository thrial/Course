import numpy as np
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import hamming
import time

# Read data
data = pd.read_csv('genres_v2.csv', dtype='unicode')

# Select columns of interest
dfSong = data[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
               'valence']]
cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
               'valence']

df = pd.DataFrame()
for col in cols:
    df[col] = dfSong[col].astype(float)

# Normalization of data
d = preprocessing.normalize(df, axis=0)
df = pd.DataFrame(d)

# Generate k vectors using Gaussian distribution
k = 1000
hash_vectors = np.random.normal(size=(k, 7))  # default mean is 0 and sd is 1

df = df.to_numpy()

all_binary_vectors = []

# Transform input vector into binary by multiplying (dot product) with hashing functions
for i in range(len(df)):
    current_binary_vector = []
    # For each vector, generate k binary bits
    for j in range(k):
        if np.dot(hash_vectors[j], df[i]) > 0:
            current_binary_vector.append(1)
        else:
            current_binary_vector.append(0)
    all_binary_vectors.append(current_binary_vector)

song_index = 5  # query point

#Start timer
start = time.process_time()

# Linear search for least Hamming distance
min_hamming_distance = hamming(all_binary_vectors[0], all_binary_vectors[song_index]) * len(all_binary_vectors[song_index])
result = []
list_hamming_distance = []
for i in range(len(df)):
    hamming_distance = hamming(all_binary_vectors[i], all_binary_vectors[song_index]) * len(all_binary_vectors[song_index])
    list_hamming_distance.append(hamming_distance)
    if i == song_index:
        continue
    if hamming_distance < min_hamming_distance:
        result=[i]
    elif hamming_distance == min_hamming_distance:
        result.append(i)
    else:
        continue

# Index of songs with least hamming distance will be printed
print(result)

print(time.process_time() - start)
print(time.process_time(), start)
