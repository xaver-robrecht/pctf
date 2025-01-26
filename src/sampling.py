import tensorflow as tf
from tqdm import tqdm

class RandomProjection():
    """Generates a random linear projection as used in @see https://scikit-learn.org/1.5/modules/random_projection.html#gaussian-random-projection
    """
    def __init__(self,dim_domain, dim_range):
        self.T = tf.random.normal(
            (dim_range,dim_domain),
            mean=0.0,
            stddev=1.0/(dim_range),
            dtype=tf.dtypes.float32,
            seed=None,
            name=None
        )
    def apply(self,x):
        return tf.transpose(tf.linalg.matmul(self.T,x,transpose_b=True))

def subsample(mb,n_sample,idx0=None):
    """Greedy selects n_samples of a given set mb by recursively selecting the elemet
     that is furthest apart from the already selected. starts with element at set[idx0,:].
     Returns the sampled indices"""
    if idx0 is None:
        idx0 =tf.random.uniform((),minval=0,maxval=n_sample,dtype=tf.int64)
    idx = tf.Variable(tf.fill([n_sample],tf.cast(-1,tf.int64)),dtype=tf.int64)
    idx[0].assign(idx0)
    idx = _subsample_tf(mb,n_sample,idx)
    return idx

def _subsample_tf(mb, n_sample,idx):
  dists = _get_rel_dists(mb,tf.gather(mb,idx[0]))
  for i in tqdm(range(1,n_sample),desc=f"subsampling memorybank of shape {mb.shape}",leave=False):
      idxi,dists = _sample_step(mb,tf.gather(mb,idx[i-1]),dists)
      idx[i].assign(idxi)
  return idx

@tf.function(jit_compile=True)
def _sample_step(mb,candidate,old_dists):
    diffs = tf.math.subtract(mb,candidate)
    sqrd_diffs = tf.math.pow(diffs,2)
    rel_dists = tf.math.reduce_sum(sqrd_diffs,1)
    new_dists= tf.reduce_min(tf.stack([rel_dists,old_dists]),axis=0)
    sel_idx = tf.argmax(new_dists)
    return sel_idx,new_dists

@tf.function(jit_compile=True)
def _get_rel_dists(mb,candidate):
    diffs = tf.math.subtract(mb,candidate)
    sqrd_diffs = tf.math.pow(diffs,2)
    rel_dists = tf.math.reduce_sum(sqrd_diffs,1)
    return rel_dists