
from sklearn.datasets import fetch_mldata
import numpy as np

# Load the dataset
dataset = fetch_mldata("MNIST original")

# Extract the features and labels
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

np.save('digit_features',features)
np.save('digit_labels',labels)