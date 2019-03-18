import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

fig, axes = plt.subplots(1, 4, figsize = (7,3))

for img, label, ax in zip(x_train[:4], y_train[:4], axes):
    ax.set_title(label)
    ax.imshow(img)
    ax.axis('off')

print(f'train images: {x_train.shape}')
print(f'train labels: {y_train.shape}')
print(f'train images: {x_test.shape}')
print(f'train labels: {y_test.shape}')

x_train = x_train.reshape(60000, 28 * 28) / 255
x_test = x_test.reshape(10000, 28 * 28) / 255

with tf.Session() as sesh:
    y_train = sesh.run(tf.one_hot(y_train, 10))
    y_test = sesh.run(tf.one_hot(y_test, 10))

learning_rate = 0.01
epochs = 10
batch_size = 100
batches = int(x_train.shape[0] / batch_size)

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(0.05 * np.random.randn(28 * 28, 10).astype(np.float32))
B = tf.Variable(0.05 * np.random.randn(10).astype(np.float32))

pred = tf.nn.softmax(tf.add(tf.matmul(X,W), B))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

x = np.linspace(1/100, 1, 100)

a = np.log([[0.04, 0.13, 0.96, 0.12],
            [0.01, 0.93, 0.06, 0.07]])

b = np.array([[0, 0, 1, 0],
            [1, 0, 0, 0]])

r_sum = np.sum(-a * b, axis=1)

r_mean = np.mean(r_sum)

print(r_sum)
print(r_mean)

with tf.Session() as sesh:
    tf_sum = sesh.run(-tf.reduce_sum(a*b, axis=1))
    tf_mean = sesh.run(tf.reduce_mean(tf_sum))

print(f' sum = {tf_sum}')
print(f' mean = {tf_mean:.5f}')

with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(batches):
            offset = i * epoch
            x = x_train[offset: offset + batch_size]
            y = y_train[offset: offset + batch_size]

            sesh.run(optimizer, feed_dict={X: x, Y: y})
            c = sesh.run(cost, feed_dict={X:x, Y:y})
        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.4f}')
    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y,1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ac = acc.eval({X: x_test, Y: y_test})
    print(ac)

    fig, axes = plt.subplots(1, 10, figsize=(8,4))
    for img, ax in zip(x_test[:10], axes):
        guess = np.argmax(sesh.run(pred, feed_dict={X: [img]}))
        ax.set_title(guess)
        ax.imshow(img.reshape((28,28)))
        ax.axis('off')

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
x = np.arange(30)
fig, ax = plt.subplots(1, figsize=(4.7, 3))
ax.plot(x, softmax(x), label='Softmax')
ax.legend()
plt.show()
