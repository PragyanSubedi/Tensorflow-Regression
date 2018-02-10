import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Adding noise to input data
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

#plt.plot(x_data, y_label, '*')

m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0
for x,y in zip(x_data,y_label):
    y_hat = m *x +b
    # Quadratic Cost function
    error += (y-y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf. global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 1000
    for i in range(training_steps):
        sess.run(train)
        final_slope, final_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope * x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label, '*')
plt.show()
