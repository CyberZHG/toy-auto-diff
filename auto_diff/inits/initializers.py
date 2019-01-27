import numpy as np
import scipy.stats


__all__ = [
    'zeros', 'ones', 'constants',
    'random_normal', 'random_uniform', 'truncated_normal',
    'glorot_normal', 'glorot_uniform',
    'lecun_normal', 'lecun_uniform',
    'he_normal', 'he_uniform',
]


def zeros(shape) -> np.ndarray:
    return np.zeros(shape)


def ones(shape) -> np.ndarray:
    return np.ones(shape)


def constants(value=0):
    def _constants(shape) -> np.ndarray:
        return np.ones(shape) * value
    return _constants


def random_normal(mean=0.0, stddev=0.05):
    def _random_normal(shape):
        return np.random.normal(loc=mean, scale=stddev, size=shape)
    return _random_normal


def random_uniform(low=-0.05, high=0.05):
    def _random_uniform(shape):
        return np.random.uniform(low=low, high=high, size=shape)
    return _random_uniform


def truncated_normal(loc, scale, lower, upper):
    def _truncated_normal(shape) -> np.ndarray:
        weights = scipy.stats.truncnorm.rvs(
            (lower - loc) / scale,
            (upper - loc) / scale,
            loc=loc,
            scale=scale,
            size=np.prod(shape),
        )
        return np.reshape(weights, newshape=shape)
    return _truncated_normal


def glorot_normal(shape) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[-1]
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    limit = 2.0
    return truncated_normal(0.0, scale, -limit, limit)(shape)


def glorot_uniform(shape) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[-1]
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    limit = np.sqrt(3.0) * scale
    return np.random.uniform(low=-limit, high=limit, size=shape)


def lecun_normal(shape) -> np.ndarray:
    scale = np.sqrt(1.0 / shape[0])
    limit = 2.0
    return truncated_normal(0.0, scale, -limit, limit)(shape)


def lecun_uniform(shape) -> np.ndarray:
    scale = np.sqrt(1.0 / shape[0])
    limit = np.sqrt(3.0) * scale
    return np.random.uniform(low=-limit, high=limit, size=shape)


def he_normal(shape) -> np.ndarray:
    scale = np.sqrt(2.0 / shape[0])
    limit = 2.0
    return truncated_normal(0.0, scale, -limit, limit)(shape)


def he_uniform(shape) -> np.ndarray:
    scale = np.sqrt(2.0 / shape[0])
    limit = np.sqrt(3.0) * scale
    return np.random.uniform(low=-limit, high=limit, size=shape)
