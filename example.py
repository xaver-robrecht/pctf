from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from src.patchcore import PatchCoreClassifier
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from src.feature_extractor import create_feature_extractor
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.sampling import RandomProjection



def load_and_preprocess_image(file_path, label=None,size = [384, 384]):
    image = tf.image.decode_png(tf.io.read_file(file_path), channels=3)
    image = tf.cast(image,dtype=tf.float32)
    image = tf.image.resize(image, size)
    return image

def create_dataset_from_dir(path_to_dir, include_label=False):
    image_files = []
    for root, dirs, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))

    file_ds = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = file_ds.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    if include_label:
        labels = np.array(["good" not in fn for fn in image_files],dtype=int)
        label_ds = tf.data.Dataset.from_tensor_slices( labels)
        dataset = tf.data.Dataset.zip(dataset,label_ds)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


DATADIR="/home/xaver/Downloads/mvtecad"

p = [1,1]
layers_to_extract=[3,4]
custom_model = create_feature_extractor(
    blocks_to_extract=layers_to_extract,
    model=tf.keras.applications.EfficientNetV2S(),
    aggregation_size=p,
    pattern = "block"
)
print(custom_model.summary())
for dataset in [os.path.join(DATADIR, x) for x in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, x))]:
    if  not "hazelnut" in dataset:
        continue
    input_data = create_dataset_from_dir(os.path.join(dataset, "train"))
    input_data=input_data.batch(8)
    input_data=input_data.take(4)
    pc = PatchCoreClassifier(custom_model,preprocessing= tf.keras.applications.efficientnet_v2.preprocess_input)
    for input_elem in tqdm(input_data,"Filling memory bank",leave=False):
        pc.add_to_memory_bank(input_elem)
    rp = RandomProjection(pc.memory_bank.shape[1],64)

    pc.subsample_memory_bank(int(round(pc.memory_bank.shape[0]*0.1)),rp.apply)
    del rp
    input_data = create_dataset_from_dir(os.path.join(dataset, "test"),include_label=True)
    pbar = tqdm(total=len(list(input_data)),desc="Evaluation progress",leave = False)
    input_data=input_data.batch(4)
    im_scores=[]
    labels = []
    for input_elem,label in input_data:
        im_score,pxl_score = pc.score(input_elem,k=10)
        im_scores += [*list(im_score)]
        labels += [*label]
        pbar.update(input_elem.shape[0])
    pbar.close()
    AUROC = roc_auc_score(labels,im_scores)
    datasetname = dataset.split('\\')[-1]
    print(f"{datasetname} \t {AUROC}\n")