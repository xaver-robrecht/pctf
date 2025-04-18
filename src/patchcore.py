import tensorflow as tf
from src.sampling import subsample 
from src.distance import k_squared_distances,squared_distances
import numpy as np

def gaussian_filter(image_tensor,sigma,n=4):
    """Applies a gaussian filter with standard deviation sigma to all images in image_tensor.

    Args:
        image_tensor (array-like,float): 4D Tensor of shape (batchsize,height,width,channels)
        sigma (float): standard deviation of gaussian filter kernel
        n (int, optional): Number of standdard deviation included in the Kernel. Defaults to 4.

    Returns:
        tensor: filtered images
    """
    kernel_len = round(n*sigma)
    x,y = tf.meshgrid(tf.range(-kernel_len,kernel_len+1,dtype=image_tensor.dtype)
                        , tf.range(-kernel_len,kernel_len+1,dtype=image_tensor.dtype))
    kernel = tf.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    kernel = tf.reshape(kernel,(*kernel.shape,1,1))
    padded = tf.pad(image_tensor, paddings=[[0,0], [kernel_len,kernel_len], [kernel_len,kernel_len], [0,0] ], mode='REFLECT')
    return tf.conv(padded,kernel,padding="VALID",strides = 4*[1])

class PatchCoreClassifier():
    def __init__(self,feature_extractor,preprocessing = lambda x:x):
        self.feature_extractor = feature_extractor
        self.memory_bank = None
        self.preprocessing=preprocessing

    def add_to_memory_bank(self,input_tensor):
        input_tensor = self.preprocessing(input_tensor)
        if self.memory_bank is None:
            model_output = self.feature_extractor(input_tensor)
            feature_vectors = tf.reshape(model_output,(-1,model_output.shape[-1]))
            self.memory_bank = feature_vectors
        else:
            model_output = self.feature_extractor(input_tensor)
            feature_vectors = tf.reshape(model_output,(-1,model_output.shape[-1]))
            self.memory_bank = tf.concat([self.memory_bank ,feature_vectors],axis=0)

    def subsample_memory_bank(self,n_sample,projection_to_reduce_dims=None,idx0=None):
        if projection_to_reduce_dims:
            idx = subsample(projection_to_reduce_dims(self.memory_bank),n_sample,idx0)
        else:
            idx = subsample(self.memory_bank,n_sample,idx0)
        self.memory_bank = tf.gather(self.memory_bank,idx,axis=0)

    def save_memory_bank(self,file_name):
        np.save(file_name,self.memory_bank,allow_pickle=False)

    def load_memory_bank(self,file_name):
        self.memory_bank=np.load(file_name,allow_pickle=False).copy()

    def score(self,input_tensor,k=10,sigma=4):
        input_tensor = self.preprocessing(input_tensor)
        model_output = self.feature_extractor(input_tensor)
        image_scores,pixel_scores = _scoring_function(model_output,k,self.memory_bank)
        pixel_scores = tf.image.resize(pixel_scores[:,:,:,tf.newaxis],input_tensor.shape[-3:-1])
        pixel_scores = tf.reshape(gaussian_filter(pixel_scores,sigma),[*pixel_scores.shape[:-1],1])
        return image_scores,pixel_scores

@tf.function
def _scoring_function(model_output,k,mb):
    input_features = tf.reshape(model_output, (-1,model_output.shape[-1]))
    dists_to_mb = tf.sqrt(squared_distances(fv=input_features,mb=mb))

    #pixelscores = distance of each feature vector to memory bank
    pixel_scores = tf.reshape(dists_to_mb,(model_output.shape[0],-1))
    
    #softmax over k closest elements of most anomal pixel
    idx_max = tf.math.argmax(pixel_scores,axis=-1)
    input_features = tf.reshape(model_output, (model_output.shape[0],-1,model_output.shape[-1]))
    input_features = tf.gather(input_features, idx_max, axis=1, batch_dims=1)
    topk_dists = tf.sqrt(k_squared_distances(
                fv=input_features,
                mb=mb,k=k))
    topk_dists = tf.reshape(topk_dists,(model_output.shape[0],k))
    image_scores = (
        1-tf.gather(tf.nn.softmax(topk_dists), indices=0, axis=-1)
        )*tf.gather(topk_dists, indices=0, axis=-1)
        
    #reshape & resize & smooth
    pixel_scores = tf.reshape(pixel_scores,model_output.shape[:-1])
    return image_scores,pixel_scores