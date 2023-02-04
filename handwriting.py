import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

# load data
digits = datasets.load_digits()
print(digits.target)

# plot one value
plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

# train model
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# visualize centroids
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2,5,1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

# test my own numbers
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.80,1.52,0.00,0.00,0.00,0.00,0.00,0.00,7.61,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.61,3.80,0.00,0.00,0.00,0.00,0.00,0.00,7.46,4.18,0.00,0.00,0.00,0.00,0.00,0.00,6.85,4.57,0.00,0.00,0.00,0.00,0.00,0.00,6.40,4.11,0.00,0.00,0.00,0.00,0.00,0.00,0.38,0.15,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,2.89,3.05,3.05,1.21,0.00,0.00,0.00,0.84,7.46,7.61,7.61,5.63,0.00,0.00,0.00,0.00,0.00,0.00,5.25,6.78,0.00,0.00,0.00,0.00,0.00,1.81,7.55,4.41,0.00,0.00,0.00,0.00,1.58,7.07,6.93,0.75,0.00,0.00,0.00,3.34,7.54,7.61,7.61,6.85,0.00,0.00,0.00,1.97,4.11,2.89,2.59,3.05,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.76,6.69,6.85,6.70,3.41,0.00,0.00,0.00,0.30,4.33,4.33,7.22,6.85,0.00,0.00,0.00,0.30,6.77,7.61,7.62,4.23,0.00,0.00,0.00,0.00,3.41,5.39,7.53,4.86,0.00,0.00,0.00,0.61,1.52,2.13,6.39,6.86,0.00,0.00,0.00,4.73,7.62,7.62,7.39,3.40,0.00,0.00,0.00,0.61,1.52,1.52,0.23,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,4.19,1.06,2.66,6.09,0.00,0.00,0.00,0.91,7.62,3.04,3.80,7.62,0.00,0.00,0.00,1.67,7.62,2.35,3.80,7.62,0.00,0.00,0.00,2.58,7.62,4.41,5.94,7.62,1.37,0.00,0.00,3.57,7.62,7.54,7.24,7.62,1.37,0.00,0.00,0.83,2.59,0.45,3.80,7.62,0.00,0.00,0.00,0.00,0.00,0.00,3.72,7.62,0.45]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(3, end='')
  elif new_labels[i] == 1:
    print(0, end='')
  elif new_labels[i] == 2:
    print(8, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(9, end='')
  elif new_labels[i] == 5:
    print(2, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(7, end='')
  elif new_labels[i] == 8:
    print(6, end='')
  elif new_labels[i] == 9:
    print(5, end='')


