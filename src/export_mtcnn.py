"""
    serve_web.py
    2018.06.19
"""

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import facenet2
import align.detect_face

def export_graph(sess, filepath, output_names, input_graph_def):
    # convert batch norm node
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    # freeze graph
    freeze_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_names
    )

    with tf.gfile.GFile(filepath, 'wb') as fp:
        fp.write(freeze_graph_def.SerializeToString())


class MTCNN:
    def __init__(self, sess):
        self.session = sess
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)

        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        self.margin = 32
        self.image_size = 160

    def detect(self, img):
        with self.session.as_default():
            if img.ndim<2:
                return [], []
            if img.ndim == 2:
                img = facenet2.to_rgb(img)
            img = img[:,:,0:3]

            bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            nrof_faces = bounding_boxes.shape[0]
            detected_faces = []
            detected_bb = []
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces>1:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-self.margin/2, 0)
                    bb[1] = np.maximum(det[1]-self.margin/2, 0)
                    bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                    detected_faces.append(scaled)
                    detected_bb.append(bb)
            return detected_faces, detected_bb

if __name__ == '__main__':
    with tf.Session() as sess:
        mtcnn = MTCNN(sess)
        export_graph(sess, 'mtcnn.pb',
            ['pnet/conv4-2/BiasAdd', 'pnet/prob1', 'rnet/conv5-2/conv5-2', 'rnet/prob1', 'onet/conv6-2/conv6-2', 'onet/conv6-3/conv6-3', 'onet/prob1'],
            sess.graph.as_graph_def())
