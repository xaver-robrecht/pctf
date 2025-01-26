from src import squared_distances, k_squared_distances
from scipy.spatial.distance import cdist
import numpy as np

def test_squared_distances():
    mb = np.random.random((100,128))
    fv = np.random.random((10,128))
    dists = cdist(fv,mb,metric="sqeuclidean").min(axis=-1)
    assert np.abs(dists-np.asarray(squared_distances(mb,fv))).max()<1e-6

def test_k_squared_distances():
    mb = np.random.random((100,128))
    fv = np.random.random((10,128))
    k=10
    test_val = np.asarray(k_squared_distances(mb=mb, fv=fv, k=k))
    ref_val = np.sort(cdist(fv,mb,metric="sqeuclidean"),axis=-1)[:,:k]
    assert np.abs(test_val-ref_val).max()<1e-6