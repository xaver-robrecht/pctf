import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf


def segment(image,scores,fnms,pxl_th):
    cmps = [plt.cm.viridis(plt.Normalize()(score)) for score in scores]
    cmps = np.array(cmps)[...,0,:3]
    for im,cmp,fn in zip(image,cmps,fnms):
        A = np.array(im)
        B = cmp
        B[B>pxl_th] = 0
        A = np.uint8(np.round(255*(A-A.min())/(A.max()-A.min())))
        B = np.uint8(np.round(255*(B-B.min())/(B.max()-B.min())))
        A = Image.fromarray(A).convert("RGB")
        B = Image.fromarray(B).convert("RGB")
        Image.blend(A,B,0.5).save(fn)