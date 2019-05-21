
### Thanks to UNI-LJ FE LST







### !nvidia-smi
### !free-m


import os

# Chooses GPU ID FOR COMPUATION
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import TF after setting environment paramaters
import tensorflow as tf

# allow_growth to not consume all GPU memory, but only what is "needed"
config = tf.ConfigProto()  # TODO: UPGRADE is: config = tf.ConfigProto(log_device_placement=True,device_count = {'GPU': 1})
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# with tf.device to choose device
# !!! IF U USE: device_count, there is no need for tf.device**
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.


# sess = tf.Session(config=config)
# Test:
print(session.run(c))

