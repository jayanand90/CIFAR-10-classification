import tensorflow as tf
import numpy as np

def normalize(x):
    """
    Normalize input to be in the range of 0 to 1 and return
    x - input image of shape (32, 32, 3)
    """
    return (x/255)

def one_hot_encode(x):
    """
    One hot encode a list of labels and return
    x - list of labels
    """

    n = np.array([0,1,2,3,4,5,6,7,8,9])
    l0 = []
    for label in x:
        tmp = (label==n).astype(np.int_)
        l0.append(tmp)
    return (np.array(l0))


def cnn_input_img(image_shape):
    """
    Return a tensorflow placeholder for input image
    image_shape: input image's shape
    """
    x = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]],name='x')
    return x


def cnn_input_label(n_classes):
    """
    Return a tensorflow placeholder for input label
    n_classes - number of classes
    """
    y = tf.placeholder(tf.float32, [None,n_classes], name='y')
    return y


def cnn_keep_prob():
    """
    Return a tensorflow placeholder for keep probability
    """
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return keep_prob

def conv_and_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Return a tensor that combines operations of convolution and max pooling
    x - input tensor
    """ 
    weights = tf.Variable(tf.random_normal([conv_ksize[0],conv_ksize[1],int(x.shape[3]),conv_num_outputs],stddev=0.05))
    biases  = tf.Variable(tf.constant(0.01,shape=[conv_num_outputs]))
    x_conv  = tf.nn.conv2d(x,weights,strides=[1,conv_strides[0],conv_strides[1],1],padding='SAME')
    x_conv  = tf.nn.bias_add(x_conv,biases)
    x_conv  = tf.nn.relu(x_conv)
    x_conv_mp = tf.nn.max_pool(x_conv,ksize=[1,pool_ksize[0],pool_ksize[1],1], strides=[1,pool_strides[0],pool_strides[1],1],padding='SAME')
    
    return x_conv_mp 

def flatten(x):
    """
    Return flattened x of dimensions (Batch Size, Flattened Image Size)
    x: A tensor (Batch Size, a,b,c), where a,b,c are the image dimensions.
    """
    return (tf.placeholder(tf.float32, [None,int(x.shape[1])*int(x.shape[2])*int(x.shape[3])]))

def fully_connected(x, num_outputs):
    """
    Add a fully connected layer to x and return
    x: 2D tensor whose first dim is batch size.
    num_outputs: desired output layer size
    """
    weights_fc = tf.Variable(tf.random_normal([int(x.shape[1]),num_outputs],stddev=0.05))
    biases_fc  = tf.Variable(tf.random_normal([num_outputs]))
    fc = tf.add(tf.matmul(x,weights_fc),biases_fc)
    fc = tf.nn.relu(fc)
    return(fc)

def output_layer(x, num_outputs):
    """
    Add a fully connected layer to x and return
    x: 2D tensor whose first dim is batch size.
    num_outputs: desired output layer size
    """
    weights_op = tf.Variable(tf.random_normal([int(x.shape[1]),num_outputs],stddev=0.05))
    biases_op  = tf.Variable(tf.random_normal([num_outputs]))
    return (tf.add(tf.matmul(x,weights_op),biases_op)) 

def convolutional_neural_network(x, keep_prob):
    """
    Create a CNN model and return logits 
    x - input tensorflow placeholder
    keep_prob - dropout keep_prob tensorflow placeholder
    """
    conv_mp1 = conv_and_maxpool(x,100,(2,2),(1,1),(2,2),(2,2))
    conv_mp2 = conv_and_maxpool(conv_mp1,400,(2,2),(1,1),(2,2),(2,2))
    cnn_flatten = tf.reshape(conv_mp2, [-1,int(flatten(conv_mp2).shape[1])])
    fc1 = fully_connected(cnn_flatten, 3072)
    fc2 = fully_connected(fc1, 1024)
    cnn_dropout = tf.nn.dropout(fc2,keep_prob)
    cnn_output = output_layer(cnn_dropout, 10)
    return (cnn_output)


tf.reset_default_graph()

# Inputs
x = cnn_input_img((32, 32, 3))
y = cnn_input_label(10)
keep_prob = cnn_keep_prob()

# Create CNN model
logits = convolutional_neural_network(x, keep_prob)
logits = tf.identity(logits, name='logits')

# Calculate cross entropy loss and use Adam Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Get model accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Function for traning the CNN on batches of images, labels
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Function for printing loss and accuracy information
    """
    loss = session.run(cost, feed_dict={
    x: feature_batch,
    y: label_batch,
    keep_prob: 1.})
    
    valid_acc = session.run(accuracy, feed_dict={
    x: valid_features,
    y: valid_labels,
    keep_prob: 1.})
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    
# Hyperparameters
epochs = 20
batch_size = 256
keep_probability = 0.8

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            # load_training_batch gets batches of input images and labels
            for batch_features, batch_labels in load_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
