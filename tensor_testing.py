import tensorflow as tf
import time 



st = time.time()
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2, 3], dtype = tf.float32)
print(rank_1_tensor)    

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype = tf.float32)
print(rank_2_tensor)

print(rank_2_tensor.shape)

print(tf.transpose(rank_2_tensor))

mat1 = tf.random.uniform((10000, 10000), -1, 1, dtype=tf.float32) 

mat2 = tf.random.uniform((10000, 10000), -1, 1, dtype=tf.float32) 
# print(f" dot: {tf.tensordot(rank_2_tensor, rank_1_tensor, 1)}")

print(f" dot: {tf.tensordot(mat1, mat2, 1)}")

print(f"time = {time.time() - st}")