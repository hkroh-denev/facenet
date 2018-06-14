"""
    clustering.py
    2018.06.11
"""

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import facenet

from scipy import misc
from sklearn.cluster import KMeans

class FaceNet:
    def __init__(self, sess, args):
        self.session = sess
        facenet.load_model(args.model)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def preprocess(self, images):
        processed_images = []
        for img in images:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = facenet.prewhiten(img)
            processed_images.append(img)
        return processed_images

    def extract_feature(self, images):
        emb_array = self.session.run(self.embeddings, feed_dict={self.images_placeholder:images, self.phase_train_placeholder:False})
        return emb_array

class FaceClustering:
    def __init__(self, num_clusters):
        self.kmeans = KMeans(n_clusters = num_clusters, random_state=0)

    def clustering(self, x):
        self.kmeans.fit(x)
        return self.kmeans.labels_

def main(args):
    print('Creating networks and loading parameters')

    input_images = facenet.load_data(facenet.get_image_paths(args.input_dir), do_random_crop=False, do_random_flip=False, image_size=args.image_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        print('Loading feature extraction model')
        feature_net = FaceNet(sess, args)

        batch_count = len(input_images) // args.batch_size
        if len(input_images) % args.batch_size != 0:
            batch_count += 1

        all_embeddings = []
        for i in range(batch_count):
            batch_images = input_images[i*args.batch_size:(i+1)*args.batch_size]
            embeddings = feature_net.extract_feature(batch_images)
            if all_embeddings == []:
                all_embeddings = embeddings
            else:
                all_embeddings = np.concatenate([all_embeddings, embeddings])

    face_cluster = FaceClustering(num_clusters=args.num_clusters)
    labels = face_cluster.clustering(all_embeddings)
    for i in range(len(labels)):
        if not os.path.exists(os.path.join(args.output_dir, str(labels[i]))):
            os.mkdir(os.path.join(args.output_dir, str(labels[i])))
        misc.imsave(os.path.join(args.output_dir, str(labels[i]), str(i) + '.jpg'), input_images[i])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('input_dir', type=str,
        help='Directory of input images (cropped and aligned)')
    parser.add_argument('output_dir', type=str,
        help='Directory of output images (separated by class id)')
    parser.add_argument('--num_clusters', type=int,
        help='Number of face cluster', default=5)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=10)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
