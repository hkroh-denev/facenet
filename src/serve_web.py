"""
    serve_web.py
    2018.06.05
"""

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import facenet2
import align.detect_face

from scipy import misc
import pickle

import cv2
import web
import time

from os.path import expanduser

def get_user_dir():
    user_home = expanduser("~")
    aww_home = os.path.join(user_home, 'aww')
    return aww_home


class MTCNN:
    def __init__(self, sess):
        self.session = sess
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)

        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        self.margin_rate = 0.5
        self.image_size = 120 * 2

    def detect(self, img):
        with self.session.as_default():
            if img.ndim<2:
                return [], []
            if img.ndim == 2:
                img = facenet2.to_rgb(img)
            img = img[:,:,0:3]

            bounding_boxes, f_points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            print('bounding_boxes', bounding_boxes)
            nrof_faces = bounding_boxes.shape[0]
            detected_faces = []
            detected_bb = []
            face_score = []
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                face_score = bounding_boxes[:, 4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces>1:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    margin0 = (det[2] - det[0]) * self.margin_rate
                    margin1 = (det[3] - det[1]) * self.margin_rate
                    bb[0] = np.maximum(det[0]-margin0, 0)
                    bb[1] = np.maximum(det[1]-margin1, 0)
                    bb[2] = np.minimum(det[2]+margin0, img_size[1])
                    bb[3] = np.minimum(det[3]+margin1, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                    detected_faces.append(scaled)
                    detected_bb.append(bb)

                    for j in range(10):
                        if j < 5:
                            f_points[j, i] = (f_points[j, i] - bb[0]) / (bb[2] - bb[0])
                        else:
                            f_points[j, i] = (f_points[j, i] - bb[1]) / (bb[3] - bb[1])

            return detected_faces, detected_bb, f_points, face_score

class FaceNet:
    def __init__(self, sess, args):
        self.session = sess
        self.image_size = 160
        facenet2.load_model(args.model)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def preprocess(self, images):
        processed_images = []
        for img in images:
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                print('Warning: custom resizing is used.')
                img = misc.imresize(img, (self.image_size, self.image_size), interp='bicubic')
            if img.ndim == 2:
                img = facenet2.to_rgb(img)
            img = facenet2.prewhiten(img)
            processed_images.append(img)
        return processed_images

    def extract_feature(self, images):
        emb_array = self.session.run(self.embeddings, feed_dict={self.images_placeholder:images, self.phase_train_placeholder:False})
        return emb_array

class FaceClassifier:
    def __init__(self, args):
        classifier_filename_exp = os.path.expanduser(args.classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile)

    def classify(self, embeddings):
        predictions = self.model.predict_proba(embeddings)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        return best_class_indices, best_class_probabilities

def face_alignment(img, face_size, f_point):
    desired_left_eye = (0.40, 0.40)
    desired_right_eye = (0.70, 0.40)
    right_eye_center = (f_point[0], f_point[5])
    left_eye_center = (f_point[1], f_point[6])

    eyes_center = ( (right_eye_center[0] + left_eye_center[0])/2 * img.shape[1], (right_eye_center[1] + left_eye_center[1])/2 * img.shape[0])
    diff_x = np.sqrt(np.square(right_eye_center[0] - left_eye_center[0]))
    diff_y = np.sqrt(np.square(right_eye_center[1] - left_eye_center[1]))
    if left_eye_center[1] < right_eye_center[1]:
        sign = -1.0
    else:
        sign = 1.0
    angle = np.degrees(np.arctan2(diff_y, diff_x))
    angle *= sign
    scale = np.sqrt(np.square(desired_right_eye[0] - desired_left_eye[0]) + np.square(desired_right_eye[1] - desired_left_eye[1])) / np.sqrt(np.square(right_eye_center[0] - left_eye_center[0]) + np.square(right_eye_center[1] - left_eye_center[1]))

    print(eyes_center, angle, scale)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = face_size * 0.5
    tY = face_size * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    (w, h) = (face_size, face_size)

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC), angle, scale

def create_point_face_image(img, face_size, f_point):
    point_img = img.copy()
    right_eye_center = (int(f_point[0] * img.shape[1]), int(f_point[5] * img.shape[0]))
    left_eye_center = (int(f_point[1] * img.shape[1]), int(f_point[6] * img.shape[0]))
    cv2.rectangle(point_img, (left_eye_center[0]-3, left_eye_center[1]-3), (left_eye_center[0]+3, left_eye_center[1]+3), (0, 255, 0), 3)
    cv2.rectangle(point_img, (right_eye_center[0]-3, right_eye_center[1]-3), (right_eye_center[0]+3, right_eye_center[1]+3), (0, 255, 0), 3)
    return point_img

def create_result_image(source_image, detected_bb, class_names, best_class_indices, best_class_probabilities):
    reduce_margin = 30
    source_image = np.array(source_image)
    result_image = source_image.copy()

    for i in range(len(detected_bb)):
        bb = detected_bb[i]
        cv2.rectangle(result_image, (bb[0] + reduce_margin, bb[1] + reduce_margin),
            (bb[2] - reduce_margin, bb[3] - reduce_margin), (0, 255, 0), 3)
        cv2.putText(result_image, class_names[best_class_indices[i]],
                    (bb[2] + 5, bb[3] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_image, '%.3f' % best_class_probabilities[i],
                    (bb[2] + 5, bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return result_image

class Serve(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('127.0.0.1', port))

class Index:
    def GET(self):
        web.header("Content-Type", "text/html; charset=utf-8")
        content = """<html><head></head><body>
        <h2>Face Recognition Demo</h2>
        <form method="POST" enctype="multipart/form-data" action="">
        <input type="file" name="myfile" />
        <br/>
        <input type="submit" />
        </form>"""
        content += "<h2>Parameters</h2>"
        content += "<ul>"
        content += "<li>model : " + web.args.model + "</li>"
        content += "<li>clasifier : " + web.args.classifier_filename + "</li>"
        content += "<li>MTCNN image size : " + str(web.mtcnn.image_size) + "</li>"
        content += "<li>FaceNet embedding size : " + str(web.feature_net.embedding_size) + "</li>"
        content += "<li>Registered class : " + str(web.classifier.class_names) + "</li>"
        content += "</ul>"
        content += "</body></html>"
        return content

    def POST(self):
        x = web.input(myfile={})
        if 'myfile' in x:
            fout = open(get_user_dir() + '/input/1.jpg','wb') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.
        raise web.seeother('/run/local')

class RunLocal:
    def GET(self):
        web.header("Content-Type", "text/html; charset=utf-8")
        start_time = time.time()
        input_path = get_user_dir() + '/input/1.jpg'
        try:
            img = misc.imread(input_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(input_path, e)
            return errorMessage
        source_image = img.copy()
        loaded_time = time.time()
        detected_faces, detected_bb, f_points, face_score = web.mtcnn.detect(img)
        if len(detected_faces) == 0:
            return '<html><body><p>No face is found!</p></body></html>'
        detected_time = time.time()

        detected_faces_cropped = []
        for i in range(len(detected_faces)):
            detected_faces_cropped.append(detected_faces[i][40:200, 40:200, :])

        preprocessed_faces = web.feature_net.preprocess(detected_faces_cropped)
        preprocessed_time = time.time()

        features = web.feature_net.extract_feature(preprocessed_faces)
        extracted_time = time.time()

        best_class_indices, best_class_probabilities = web.classifier.classify(features)
        classified_time = time.time()

        result_image = create_result_image(source_image, detected_bb, web.classifier.class_names, best_class_indices, best_class_probabilities)
        misc.imsave(get_user_dir() + '/output/result.png', result_image)

        for i in range(len(detected_faces)):
            misc.imsave(get_user_dir() + '/output/face-%s.png' % i, detected_faces[i])
            misc.imsave(get_user_dir() + '/output/face-%sc.png' % i, detected_faces_cropped[i])

        aligned_faces = []
        angles = []
        scales = []
        for i in range(len(detected_faces)):
            aligned_face, angle, scale = face_alignment(detected_faces[i], web.mtcnn.image_size, f_points[:, i])
            aligned_face = misc.imresize(aligned_face, (web.feature_net.image_size, web.feature_net.image_size), interp='bicubic')
            aligned_faces.append(aligned_face)
            angles.append(angle)
            scales.append(scale)

        preprocessed_aligned_faces = web.feature_net.preprocess(aligned_faces)
        features2 = web.feature_net.extract_feature(preprocessed_aligned_faces)
        best_class_indices2, best_class_probabilities2 = web.classifier.classify(features2)
        for i in range(len(detected_faces)):
            misc.imsave(get_user_dir() + '/output/face-%sa.png' % i, aligned_faces[i])
            point_face = create_point_face_image(misc.imread(get_user_dir() + '/output/face-%s.png' % i), web.mtcnn.image_size, f_points[:, i])
            misc.imsave(get_user_dir() + '/output/face-%sp.png' % i, point_face)
            point_face2 = create_point_face_image(misc.imread(get_user_dir() + '/output/face-%sa.png' % i), web.mtcnn.image_size, [0.35, 0.65, 0, 0, 0, 0.35, 0.35, 0, 0, 0])
            misc.imsave(get_user_dir() + '/output/face-%sap.png' % i, point_face2)

        s = """<html><body>
            <h2>Result image</h2>
            <p><img src="/output/result.png" width="960"/></p>
            <h2>Detected Faces</h2><p>"""
        for i in range(len(detected_faces)):
            #s += '<img src="/output/face-%sp.png" />' % i
            s += '<img src="/output/face-%sc.png" />' % i
            s += web.classifier.class_names[best_class_indices[i]] + ': '
            s += 'rec=%.3f' % (best_class_probabilities[i]) + ' det=%.3f' % (face_score[i])
            s += '<img src="/output/face-%sa.png" />' % i
            #s += '<img src="/output/face-%sap.png" />' % i
            s += web.classifier.class_names[best_class_indices2[i]] + ': '
            s += 'rec=%.3f' % (best_class_probabilities2[i]) + ', '
            s += 'angle=%.3f, scale=%.3f' % (angles[i], scales[i])
            s += '<br>'
        s += '</p><h2>Processing Time</h2><p>'
        s += 'Total time: %.5f' % (classified_time - start_time) + '<br>'
        s += 'Image load : %.5f' % (loaded_time - start_time) + '<br>'
        s += 'Face detection : %.5f' % (detected_time - loaded_time) + '<br>'
        s += 'Face preprocessing : %.5f' % (preprocessed_time - detected_time) + '<br>'
        s += 'Feature extraction : %.5f' % (extracted_time - preprocessed_time) + '<br>'
        s += 'Classifier : %.5f' % (classified_time - extracted_time)
        s += '</p></body><html>'
        return s

class Output:
    def GET(self, param):
        web.header("Content-Type", "image/png")
        f = open(get_user_dir() + '/output/' + param, 'rb')
        return f.read()

def main(args):
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            mtcnn = MTCNN(sess)

            print('Loading feature extraction model')
            feature_net = FaceNet(sess, args)

            print('Loading classifer model')
            classifier = FaceClassifier(args)

            print('Web server')
            urls = ('/', 'Index',
                    '/run/local', 'RunLocal',
                    '/output/(.+)', 'Output')
            webapp = Serve(urls, globals())
            web.args = args
            web.sess = sess
            web.mtcnn = mtcnn
            web.feature_net = feature_net
            web.classifier = classifier
            webapp.run(port=args.port_number)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default="model/20180402-114759.pb")
    parser.add_argument('--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.',
        default="model/my_classifier-1.pkl")
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser.add_argument('--port_number', type=int,
        help='Port number of web server', default=8080)
    return parser.parse_args(argv)

if __name__ == '__main__':
    user_dir = get_user_dir()
    print('user_dir:', user_dir)
    if not os.path.exists(user_dir):
        os.mkdir(user_dir)

    input_dir = os.path.join(user_dir, 'input')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    output_dir = os.path.join(user_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(parse_arguments(sys.argv[1:]))
