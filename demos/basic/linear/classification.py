import numpy as np
import auto_diff as ad


def gen_config(base_config: dict = None) -> dict:
    """Generate basic configuration."""
    config = {'input_len': np.random.randint(1, 11)}
    if base_config is not None and 'input_len' in base_config:
        config['input_len'] = base_config['input_len']
    config = {
        'input_len': config['input_len'],
        'hidden_dim': np.random.randint(10, 50),
        'batch_size': np.random.randint(32, 500),
        'learning_rate': 1e-3,
        'expected_w': (np.random.random(config['input_len']) - 0.5) * 10.0,
        'expected_b': (np.random.random() - 0.5) * 10.0,
    }
    if base_config is not None:
        for key, val in base_config.items():
            config[key] = val
    return config


def gen_linear_model(config: dict, verbose=False):
    """Generate a linear model.

    :param config: Configuration.
    :param verbose: Print loss and gradients if it is True.
    :return: Model, loss, placeholders and variables.
    """
    x = ad.placeholder(shape=(None, config['input_len']), name='X')
    y = ad.placeholder(shape=(None,), name='Y')

    w1 = ad.variable(
        initializer=ad.inits.random_normal(),
        shape=(config['input_len'], config['hidden_dim']),
        name='W1',
    )
    b1 = ad.variable(
        initializer=ad.inits.zeros,
        shape=config['hidden_dim'],
        name='b1',
    )

    v = ad.acts.leaky_relu(ad.dot(x, w1) + b1)

    w2 = ad.variable(
        initializer=ad.inits.random_normal(),
        shape=(config['hidden_dim'], 2),
        name='W2',
    )
    b2 = ad.variable(
        initializer=ad.inits.zeros,
        shape=2,
        name='b2',
    )

    y_pred = ad.acts.softmax(ad.dot(v, w2) + b2)

    loss = ad.square(y - y_pred).mean()

    if verbose:
        print('Loss:', loss)

    return y_pred, loss, [x, y], [w1, b1, w2, b2]


def data_generator(config: dict):
    """Generate data infinitely with the given parameter and batch size.

    :param config: Configuration.
    :return: Linear data for placeholders.
    """
    batch_size = config['batch_size']
    classes = np.random.randint(0, 1, (batch_size,)) - 0.5
    while True:
        batch_x = (np.random.random((batch_size, config['input_len'])) - 0.5) * 10.0
        linear_y = np.dot(batch_x, config['expected_w']) + config['expected_b'] + classes
        linear_y += (np.random.random(batch_size) - 0.5) * 0.1  # Add random noise
        batch_y = np.zeros((batch_size, 2))
        batch_y[np.arange(batch_size), (linear_y > 0.0).astype(np.int)] = 1.0
        yield batch_x, batch_y


def train_model(loss: ad.Operation, placeholders: list, variables: list, config: dict, verbose=False) -> None:
    """Train the linear model with SGD.

    :param loss: Loss operation.
    :param placeholders: Placeholders of inputs.
    :param variables: Trainable variables.
    :param config: Configuration.
    :param verbose: Whether to show the losses.
    """
    sess = ad.Session()
    x, y_true = placeholders
    learning_rate = config['learning_rate']
    for step, (batch_x, batch_y) in enumerate(data_generator(config)):
        if step > 50000:
            break
        sess.prepare()
        feed_dict = {x: batch_x, y_true: batch_y}
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss.backward()
        lr = learning_rate / (1.0 + 1e-5 * step)
        for var in variables:
            var.update_add(-lr * var.gradient)
        if verbose:
            print('\rStep %d - Loss %.4f' % (step, loss_val), end='')
            if loss_val < 1e-4:
                break
    if verbose:
        print('')


def check_result(model: ad.Operation, placeholders: list, config: dict, verbose=False):
    """Check the trained model.

    :param model: The linear model.
    :param placeholders: Placeholders of inputs.
    :param config: Configuration.
    :param verbose: Whether to show the sample outputs.
    """
    sess = ad.Session()
    x, y_true = placeholders
    for batch_x, batch_y in data_generator(config):
        sess.prepare()
        feed_dict = {x: batch_x, y_true: batch_y}
        y_pred_val = sess.run(model, feed_dict=feed_dict)
        y_true_cls = np.argmax(batch_y, axis=-1)
        y_pred_cls = np.argmax(y_pred_val, axis=-1)
        accuracy = (y_true_cls == y_pred_cls).astype(np.float64).sum() / config['batch_size']
        if verbose:
            print('Expected: ', y_true_cls[:5])
            print('Actual:   ', y_pred_cls[:5])
            print('Accuracy: ', accuracy)
        assert accuracy > 0.8
        break


def main(base_config: dict = None, verbose=False):
    config = gen_config(base_config)
    model, loss, placeholders, variables = gen_linear_model(config, verbose)
    train_model(loss, placeholders, variables, config, verbose)
    check_result(model, placeholders, config, verbose)


if __name__ == '__main__':
    main(verbose=True)
