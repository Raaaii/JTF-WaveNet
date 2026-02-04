"""
This is a module for additional tensorflow function and layers that we need.
"""

import tensorflow as tf


@tf.function
def tf_ft(signal):
    """
    Args:
        signal (tf.Tensor): the tensor describing the signal

    Returns:
        ft (tf.Tensor): fourier transform of the signal
    """

    return tf.signal.fft(signal)


@tf.function
def complex_to_real_reshape(signal):
    """This function takes the complex signal and splits it into a (2, N)
    tensor where the first part is the real part and the second part is the imaginary part

    Args:
        signal (tf.Tensor): (N,) shaped tensor containing the complex signal

    Returns:
        tf.Tensor: The signal seperated into a (2,N) shaped tensor
    """

    real = tf.math.real(signal)
    imag = tf.math.imag(signal)

    real = tf.cast(real, tf.dtypes.float64)
    imag = tf.cast(imag, tf.dtypes.float64)

    signal_real = tf.stack([real, imag])

    return signal_real


def real_to_complex_reshape(signal):
    """This takes the real signal with shape (2, N) and returns the complex signal
    with shape (N,)

    Args:
        signal (tf.Tensor): (2, N) real signal where the real and complex parts are seperate rows

    Returns:
        tf.Tensor: (N,) shaped complex tensor
    """

    complex_part = 1j * tf.cast(signal[1], tf.dtypes.complex64)
    real_part = tf.cast(signal[0], tf.dtypes.complex64)
    complex_signal = complex_part + real_part

    return complex_signal


def signal_to_model_target(signal):
    """Takes a signal, does the FT and then changes the shape to be in the form (2,N)

    Args:
        signal (tf.Tensor): (N,) shaped signal

    Returns:
        tf.Tensor: The reshaped fourier transform of the signal
    """

    ft = tf_ft(signal)
    ft_reshaped = complex_to_real_reshape(ft)
    return ft_reshaped


@tf.function
def complex_to_real_reshape_2d(signal):
    """This function takes the complex signal and interleaves it along the last dimension

    Args:
        signal (tf.Tensor): (..., N) shaped tensor containing the complex signal

    Returns:
        tf.Tensor: The signal seperated into a (..., 2N) shaped tensor
    """

    real = tf.math.real(signal)
    imag = tf.math.imag(signal)

    real = tf.cast(real, tf.dtypes.float64)
    imag = tf.cast(imag, tf.dtypes.float64)

    signal_real = tf.stack([real, imag], axis=2)
    signal_real = tf.reshape(signal_real, [-1, tf.shape(signal_real)[1] * 2])

    return signal_real


@tf.function
def real_to_complex_reshape_2d(signal):
    """This takes the real signal with shape (... , 2N) and returns the complex signal
    with shape (..., N)

    Args:
        signal (tf.Tensor): (..., N) complex signal

    Returns:
        tf.Tensor: (N,) shaped complex tensor
    """

    a = signal[:, 0::2]
    b = signal[:, 1::2]
    return tf.complex(a, b)
