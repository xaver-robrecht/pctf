import tensorflow as tf

class ResizeLayer(tf.keras.Layer):
    def __init__(self,upsampsize):
        super(ResizeLayer,self).__init__()
        self.upsampsize=upsampsize
        self.trainable=False
    def call(self, x):
        y = tf.concat([tf.image.resize(elem, method="bilinear", size=self.upsampsize) for elem in x],axis=-1)
        return y

def create_feature_extractor(blocks_to_extract, model, aggregation_size, pattern="block"):
    layer_names = []
    for i in range(max(blocks_to_extract) + 1):
        if i in blocks_to_extract:
            for j, lyr in enumerate(model.layers):
                if (f"{pattern}{i}" in lyr.name) and (f"{pattern}{i}" not in model.layers[j + 1].name):
                    layer_names += [lyr.name]
                    break

    #create truncated model with aggregator and resizing appended
    original_model=model
    original_model.trainable = False
    inputs = original_model.input
    model(inputs)
    intermediate_layers = [
        original_model.get_layer(name).output for name in layer_names
    ]
    upsampsize = tf.convert_to_tensor(
        original_model.get_layer(layer_names[0]).output.shape[1:3],
        dtype=tf.int32,
    )
    aggregated_layers = [tf.keras.layers.AveragePooling2D(pool_size=p_val,
                                                      strides=1,
                                                      padding="same",)(layer)
                          for p_val,layer in zip(aggregation_size,intermediate_layers)]
    concatenated_layers = ResizeLayer(upsampsize)(aggregated_layers)
    
    truncated_model = tf.keras.Model(
        inputs=inputs,
        outputs=concatenated_layers)
    truncated_model.trainable=False
    return truncated_model