import tensorflow as tf

tf.TF_ENABLE_ONEDNN_OPTS=0

with tf.device("/cpu:0"):
    x = tf.Tensor([[1, 2], [1, 2]])
    print(x)