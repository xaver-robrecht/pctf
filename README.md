# Anomaly Detection using Patchcore
This Repository contains an implementation of PatchCore(https://arxiv.org/abs/2106.08265) using (mostly) Tensorflow.
PatchCore is an anomaly detection algorithm that utilizes a large CNN trained on Imagenet to extract features of normal samples.
These features are then stored in what the authros call a *Memory Bank*. This *Memory Bank* can be reduced to a subset using various methods, of which greedy furthest point sampling is the one the authors originally select. I do plan to implement other Algorithms too, as the sampling process can take quite long on CPUs. The anomaly score of an input pixel is equal to its corresponding feature vectors distance to the coreset. To calculate the image anomaly score the authors suggested weighting the maximum pixel distance by a factor that depends on the isolation of the memory bank vector closest too the most anomal pixel - I'll go into those details later.

# Stepts to a successfull implementation
Essentially a module that implements the classifier will need to supply three functions
- a feature extractor that extracts the outputs of given layers out of a CNN, optionally applies pooling to each output and concatenates each resized feature map to a single output tensor
- a sampling function that can be used to select the coreset
- a function that returns the k closest elements in a set with respect to a given vector