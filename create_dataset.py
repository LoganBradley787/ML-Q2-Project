import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd

features = 10

x_clusters, y_clusters = make_blobs(
    n_samples=[1000, 300, 300],
    centers=[
        [0]*features,
        [3]*features,
        [-3]*features, 
    ],
    cluster_std=[3, 7, 7],             
    random_state=1
)

x_noise = np.random.normal(loc=0, scale=10, size=(300, features)) # 300 instances of noise
y_noise = np.random.choice([0, 1, 2], size=300, p=[0.4, 0.3, 0.3]) # noise assigned to random class (40% probability of class 0, 30% probability of class 1, 30% probability of class 2)


x = np.concatenate([x_clusters, x_noise]) 
y = np.concatenate([y_clusters, y_noise])
df = pd.DataFrame(x)
df["class"] = y

train, test = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=1
)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
