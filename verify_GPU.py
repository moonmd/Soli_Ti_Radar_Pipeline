import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    print("GPU is working!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("GPU computation result:\n", c.numpy())
else:
    print("GPU not detected - using CPU")
