import tensorflow as tf
from sklearn.utils import shuffle
import model
import pickle
import cv2
import numpy as np


training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
# testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

X_train_gray = []
X_valid_gray = []
X_test_gray = []
# print(X_train.shape)
for img in X_train:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_expanded = img[:, :, np.newaxis]
    X_train_gray.append(img_expanded)
X_train_gray = np.array(X_train_gray, dtype=np.float32)
for img in X_valid:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_expanded = img[:, :, np.newaxis]
    X_valid_gray.append(img_expanded)
X_valid_gray = np.array(X_valid_gray, dtype=np.float32)
# for img in X_test:
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img_expanded = img[:, :, np.newaxis]
#     X_test_gray.append(img_expanded)
# X_test_gray = np.array(X_test_gray, dtype=np.float32)
# print(X_train_gray)

X_train = X_train_gray
X_valid = X_valid_gray
# X_test = X_test_gray

X_train = ((X_train - 128.0)/128.0) - 1
X_valid = ((X_valid - 128.0)/128.0) - 1
# X_test = ((X_test - 128.0)/128.0) - 1

EPOCHS = 30
BATCH_SIZE = 128
rate = 0.001
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(tf.int32, shape=(None))
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
one_hot_y = tf.one_hot(y, 43)

# cross entropy error function is used for loss.
# Adam optimizer is used for learning as it is more robust and faster than gradient descent.
logits = model.deep_net(x, is_training, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    """
    This function defines the evaluation of the given validation or testing data on the learned model.
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, is_training: False, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True, keep_prob: 0.75})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './deep_net')
    print("Model saved")