import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Global variables
NUM_LABELS = 2   # The number of labels in a binary classification problem is 2
BATCH_SIZE = 100 # The number of training examples to use per training step

# Define the flags useable from the command line
tf.app.flags.DEFINE_string('train', None, 'File containing the training data')
tf.app.flags.DEFINE_string('test', None, 'File containing the test data')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')

FLAGS = tf.app.flags.FLAGS

# Extract the numpy representations of the labels and features from
# a CSV file. Assume each row consists of:
#
#       label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    # Arrays to hold the labels and feature vectors
    X = [] # Features
    y = [] # Labels

    # Iterate over the rows and split the label from the features.
    # Then convert labels to integers and features to floating point values.
    file = open(filename, "r")
    for line in file:
        row = line.strip().split(",")
        y.append(int(row[0]))
        X.append([float(x) for x in row[1:]])

    # Convert the array of floats into a numpy float matrix
    X_np = np.matrix(X).astype(np.float32)

    # Convert the labels into a numpy array
    y_np = np.array(y).astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot encoded matrix
    y_one_hot = (np.arange(NUM_LABELS) == y_np[:, None]).astype(np.float32)

    # Return a tuple of features and labels
    return X_np, y_one_hot


# Be verbose?
verbose = FLAGS.verbose

# Get the data
train_data_filename = FLAGS.train
test_data_filename  = FLAGS.test

# Extract data into numpy matrices
train_data, train_labels = extract_data(train_data_filename)
test_data, test_labels   = extract_data(test_data_filename)

# Get the shape of the training data
train_size, num_features = train_data.shape

# Get the number of epochs for training
num_epochs = FLAGS.num_epochs

# This is where the training samples and labels are fed into the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
X = tf.placeholder(tf.float32, shape=(None, num_features))
y = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))

# For the test data, hold the entire dataset in one constant node.
test_data_node = tf.constant(test_data)

# Define and initialize the network

# These are the weights that inform how much each feature contributes to
# the classification. Here we're initializing the weights to zero, however
# many deep learning projects have found success initializing the weights
# to random values.
W = tf.Variable(tf.zeros([num_features, NUM_LABELS])) # Theta
b = tf.Variable(tf.zeros([NUM_LABELS]))               # Intercept term

# Softmax regression is a generalization of logistic regression that
# supports multiclass outputs without having to train and combine
# multiple binary classifiers together.
p = tf.nn.softmax(tf.matmul(X, W) + b)

# Optimization
cross_entropy = -tf.reduce_sum(y * tf.log(p))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(p, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a local session to run the computation.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters
    sess.run(init)

    if verbose:
        print("Initialized!")
        print()
        print("Training...")

        # Iterate an train
        for step in range(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print(step)

            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
            sess.run(train_step, feed_dict={X: batch_data, y: batch_labels})

            if verbose and offset >= train_size - BATCH_SIZE:
                print

        # Give very detailed output
        if verbose:
            print
            print('Weight Matrix.')
            print(sess.run(W))
            print()
            print('Bias Vector')
            print(sess.run(b))
            print()
            print("Applying model to first test instance.")
            first = test_data[:1]
            print("Point =", first)
            print("Wx+b =", sess.run(tf.matmul(first, W) + b))
            print("Softmax(Wx+b) =", sess.run(tf.nn.softmax(tf.matmul(first, W) + b)))
            print()

        print("Accuracy:", accuracy.eval(feed_dict={X: test_data, y: test_labels}))
