
import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import facenet
import align.detect_face

from scipy import misc
import pickle

import cv2

def main(args):
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
            minsize = 20 # minimum size of face
            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor
        
            try:
                img = misc.imread(args.input_image)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(args.input_image, e)
                print(errorMessage)
                return
            
            if img.ndim<2:
                print('Unable to align "%s"' % image_path)
                text_file.write('%s\n' % (output_filename))
                return
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:,:,0:3]
        
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
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
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    detected_faces.append(scaled)
                    detected_bb.append(bb)
                print(nrof_faces, 'faces are detected')
                print(detected_bb)
            source_image = img
                    
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            
            # preprocessing
            images = []
            for img in detected_faces:
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = facenet.prewhiten(img)
                images.append(img)            
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            
            print('Testing classifier')
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

            source_image = np.array(source_image)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            
            for i in range(len(detected_bb)):
                bb = detected_bb[i]
                cv2.rectangle(source_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
                cv2.putText(source_image, class_names[best_class_indices[i]],
                            (bb[0], bb[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(source_image, '%.3f' % best_class_probabilities[i],
                            (bb[0], bb[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite('result.png', source_image)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_image', type=str, help='Path to image which contains some faces')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')    
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)    
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
