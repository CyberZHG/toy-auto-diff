import numpy as np
import auto_diff as ad

# Basic configuration
num_x = np.random.randint(2, 11)
learning_rate = 1e-4
batch_size = np.random.randint(32, 500)


# Initialize the model
x = ad.placeholder(shape=(None, num_x), name='X')
y = ad.placeholder(shape=(None,), name='Y')

w = ad.variable(np.random.random(num_x), name='W')
b = ad.variable(0.0, name='b')

y_pred = ad.dot(x, w) + b
loss = ad.square(y - y_pred).sum() / batch_size

print('Loss:', loss)

loss.backward()
print('Gradient for W:', w.gradient)
print('Gradient for b:', b.gradient)


# Initialize the target and linear data generator
expected_w = (np.random.random(num_x) - 0.5) * 10.0
expected_b = (np.random.random() - 0.5) * 10.0


def data_generator(batch_size=32):
    while True:
        batch_x = (np.random.random((batch_size, num_x)) - 0.5) * 10.0
        batch_y = np.dot(batch_x, expected_w) + expected_b
        batch_y += (np.random.random(batch_size) - 0.5) * 0.1
        yield batch_x, batch_y


# Train the model with SGD
sess = ad.Session()
for step, (batch_x, batch_y) in enumerate(data_generator(batch_size)):
    if step > 50000:
        break
    sess.prepare()
    feed_dict = {x: batch_x, y: batch_y}
    loss_val = sess.run(loss, feed_dict=feed_dict)
    w_grad = sess.run(w.gradient, feed_dict=feed_dict)
    b_grad = sess.run(b.gradient, feed_dict=feed_dict)
    w.update_add(-learning_rate * w_grad)
    b.update_add(-learning_rate * b_grad)
    print('Step %d - Loss %.4f' % (step, loss_val), end='\r')
    if loss_val < 1e-4:
        break

print('')
for batch_x, batch_y in data_generator(32):
    sess.prepare()
    feed_dict = {x: batch_x, y: batch_y}
    y_pred_val = sess.run(y_pred, feed_dict=feed_dict)
    print('Expected: ', batch_y[:5])
    print('Actual:   ', y_pred_val[:5])
    assert np.alltrue(batch_y - y_pred_val < 1.0)
    break
