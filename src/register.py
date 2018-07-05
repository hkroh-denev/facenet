import os
import sys
import argparse

import tensorflow as tf
import numpy as np

import facenet2
import align.detect_face

from scipy import misc
import pickle


import cv2

print(cv2.__version__)

def cam_test():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        cv2.imshow('video capture', frame)
        ch = 0xff & cv2.waitKey(1)
        if ch == 27:
            break

class MTCNN:
    def __init__(self, sess):
        self.session = sess
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)

        self.minsize = 80 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        self.image_size = 120 * 2
        self.margin_rate = 0.5  # margin = detected_box_size * margin_rate

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
                #if nrof_faces>1: #??
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

def face_alignment(img, face_size, f_point):
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
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

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

def extract_face_images(args):

    video_ext = ['.mkv', '.mp4', '.avi', '.wmv']
    image_ext = ['.jpg', '.jpeg', '.png', '.gif']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        mtcnn = MTCNN(sess)

        count = 0

        if os.path.isdir(args.input):
            input_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
        else:
            input_files = [args.input]

        print(input_files)

        input_index = 0
        total_faces = 0
        total_rejected = 0
        for input_file in input_files:
            filename, file_ext = os.path.splitext(input_file)
            print(input_file, file_ext)
            if file_ext in video_ext:
                vidcap = cv2.VideoCapture(input_file)
                success, image = vidcap.read()
                success = True
                while success:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * args.interval))
                    pos_msec = vidcap.get(cv2.CAP_PROP_POS_MSEC)
                    if count * args.interval - pos_msec > args.interval:
                        break;
                    success, image = vidcap.read()
                    if success:
                        detected_faces, detected_bb, f_points, face_score = mtcnn.detect(image)
                        rejected = 0
                        for i in range(len(detected_faces)):
                            if face_score[i] > 0.8:
                                aligned_face, angle, scale = face_alignment(detected_faces[i], mtcnn.image_size, f_points[:, i])
                                aligned_face = misc.imresize(aligned_face, (args.image_size, args.image_size), interp='bicubic')
                                cv2.imwrite(args.output + '/%s_%03d_%08d_%03d.jpg' % (args.filename_prefix, input_index, count, i), aligned_face)
                            else:
                                rejected += 1
                        print(count * args.interval // 1000, 'face:', len(detected_faces), 'rejected:', rejected)
                        total_faces += len(detected_faces)
                        total_rejected += rejected
                    count += 1
            elif file_ext in image_ext:
                image = cv2.imread(input_file)
                detected_faces = []
                rejected = 0
                if type(image) is np.ndarray:
                    detected_faces, detected_bb, f_points, face_score = mtcnn.detect(image)

                    for i in range(len(detected_faces)):
                        if face_score[i] > 0.8:
                            aligned_face = face_alignment(detected_faces[i], mtcnn.image_size, f_points[:, i])
                            aligned_face = misc.imresize(aligned_face, (args.image_size, args.image_size), interp='bicubic')
                            cv2.imwrite(args.output + '/%s_%03d_%08d_%03d.jpg' % (args.filename_prefix, input_index, count, i), aligned_face)
                        else:
                            rejected += 1
                    print(input_index, 'face:', len(detected_faces), 'rejected:', rejected)
                else:
                    print('invalid input file')
                total_faces += len(detected_faces)
                total_rejected += rejected
            input_index += 1
        print('total files: ', input_index)
        print('total faces:', total_faces, 'total rejected:', total_rejected)

if __name__ == '__main__':
    print('extract face from video or image file(s)')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to the input video file')
    parser.add_argument('--output', help='path to the output images')
    parser.add_argument('--interval', type=int, default=1000, help='saving interval in milliseconds')
    parser.add_argument('--image_size', type=int, default=160, help='size of output image')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser.add_argument('--filename_prefix', default='frame')
    args = parser.parse_args()
    print(args)
    extract_face_images(args)
