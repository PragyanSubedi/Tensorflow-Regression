import tensorflow as tf
import numpy as np

#wx+b=z

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32,(None,n_features))  #[anything,10]
w= tf.variable(tf_random_normal([n_features,n_dense_neurons])) # [10,3]

b= tf. variable (tf.ones([n_dense_neurons])) # [1,1,1]

xW = tf.matmul(x,w) #[1, 10] *[10,3]
z= tf.add(xW,b) # [1,3] + [1,3]
a = tf.sigmoid(z) #Activation function

init = tf.global_variables_initializer()

with tf.session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_feaures])})

print(layer_out)

