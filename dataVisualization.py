import matplotlib.pyplot as plt
import numpy as np
import pickle


training_file = "traffic-signs-data/train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
train_features = np.array(train['features'])
train_labels = np.array(train['labels'])
n_classes = len(set(train["labels"]))

for i in range(n_classes):
    for j in range(len(train_labels)):
        if i == train_labels[j]:
            print('Class: ', i)
            plt.imshow(train_features[j])
            plt.show()
            break

images_per_class = np.bincount(train_labels)
max_images = np.max(images_per_class)

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
ax.set_ylabel('Images')
ax.set_xlabel('Class')
ax.set_title('Number of images per class')
ax.bar(range(len(images_per_class)), images_per_class, 1/3, color='red', label='Images per class')
plt.show()