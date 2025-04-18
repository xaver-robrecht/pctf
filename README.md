# Anomaly Detection using PatchCore
This repository contains an implementation of [PatchCore](https://arxiv.org/abs/2106.08265) using (mostly) TensorFlow. PatchCore is an anomaly detection algorithm that utilizes a large CNN trained on ImageNet to extract features of normal samples.

These features are then stored in what the authors call a *Memory Bank*. This *Memory Bank* can be reduced to a subset using various methods, of which greedy furthest point sampling is the one the authors originally selected. I do plan to implement other algorithms too, as the sampling process can take quite long on CPUs. The anomaly score of an input pixel is equal to its corresponding feature vector's distance to the coreset. To calculate the image anomaly score, the authors suggested weighting the maximum pixel distance by a factor that depends on the isolation of the memory bank vector closest to the most anomalous pixel - I'll go into those details later.

# Steps to a Successful Implementation
Essentially, a module that implements the classifier will need to supply three functions:
- A feature extractor that extracts the outputs of given layers from a CNN, optionally applies pooling to each output, and concatenates each resized feature map into a single output tensor.
- A sampling function that can be used to select the coreset.
- A function that returns the k closest elements in a set with respect to a given vector.
# MISC
There is one detail in which this implementation differs from the procedure described in the paper when calculating the image anomaly scores. While the authors suggest aggregating over the set $\mathcal{N}_b(m^\star)$, I aggregate over $\mathcal{N}_b(m^{\text{test,} \star})$. This should not make a huge difference, as $m^{\star}$ is the nearest vector in $\mathcal{M}$ for $m^{\text{test,} \star}$. Thus, when $m^\star$ is isolated with respect to other elements of $\mathcal{M}$, so should be $m^{\text{test,} \star}$. This way, I can just use the softmax function to calculate the weighting.
