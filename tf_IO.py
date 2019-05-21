import pickle
from tqdm import tqdm
import tensorflow as tf

def save(sess, save_to = 'weights.pickle'):
    """
    :param tf.Session() sess:
    :return:
    """
    dct = {}
    for i in tqdm(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
        dct[i.name] = sess.run(i.name)
        #print(i.name)
    with open(save_to, 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(sess, load_from ='weights.pickle'):
    with open(load_from, 'rb') as handle:
        b = pickle.load(handle)
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var.name in b:
            sess.run(var.assign(b[var.name]))