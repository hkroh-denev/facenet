import os
import sys
import argparse

import tensorflow as tf
import numpy as np

import facenet2
import align.detect_face

from scipy import misc
import pickle
import tqdm

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
    def __init__(self, sess, multiple=1):
        self.session = sess
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)

        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

        self.image_size = 120 * 2
        self.margin_rate = 0.5  # margin = detected_box_size * margin_rate

        self.detect_multiple_faces = multiple

    def detect(self, img):
        with self.session.as_default():
            if img.ndim<2:
                return [], []
            if img.ndim == 2:
                img = facenet2.to_rgb(img)
            img = img[:,:,0:3]

            bounding_boxes, f_points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            #print('bounding_boxes', bounding_boxes)
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
                    if self.detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det_arr.append(det[index,:])
                else:
                    det_arr.append(np.squeeze(det))

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
                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bicubic')
                    detected_faces.append(scaled)
                    detected_bb.append(bb)

                    for j in range(10):
                        if j < 5:
                            f_points[j, i] = (f_points[j, i] - bb[0]) / (bb[2] - bb[0])
                        else:
                            f_points[j, i] = (f_points[j, i] - bb[1]) / (bb[3] - bb[1])

            return detected_faces, detected_bb, f_points, face_score

def face_alignment(img, face_size, f_point):
    desired_left_eye = (0.30, 0.40)
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

    #print(eyes_center, angle, scale)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = face_size * 0.5
    tY = face_size * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    (w, h) = (face_size, face_size)

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC), angle, scale

def detect_video(mtcnn, input_file, input_index, output_path, args):
    detected = 0
    rejected = 0
    vidcap = cv2.VideoCapture(input_file)
    success, image = vidcap.read()
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * args.interval))
        pos_msec = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        if count * args.interval - pos_msec > args.interval:
            break;
        success, image = vidcap.read()
        if success:
            detected_faces, detected_bb, f_points, face_score = mtcnn.detect(image)
            for i in range(len(detected_faces)):
                if face_score[i] > 0.8:
                    aligned_face, angle, scale = face_alignment(detected_faces[i], mtcnn.image_size, f_points[:, i])
                    aligned_face = misc.imresize(aligned_face, (args.image_size, args.image_size), interp='bicubic')
                    cv2.imwrite(os.path.join(output_path, '%03d_%08d_%03d.jpg' % (input_index, count, i)), aligned_face)
                    detected += 1
                else:
                    rejected += 1
        count += 1

    return detected, rejected

def detect_image(mtcnn, input_file, input_index, output_path, args):
    detected = 0
    rejected = 0
    image = cv2.imread(input_file)
    if type(image) is np.ndarray:
        detected_faces, detected_bb, f_points, face_score = mtcnn.detect(image)

        for i in range(len(detected_faces)):
            if face_score[i] > 0.0:
                aligned_face, angle, scale = face_alignment(detected_faces[i], mtcnn.image_size, f_points[:, i])
                aligned_face = misc.imresize(aligned_face, (args.image_size, args.image_size), interp='bicubic')
                if args.preserve_file_name != 0:
                    #print(input_file, '=>', os.path.join(output_path, os.path.basename(input_file)))
                    cv2.imwrite(os.path.join(output_path, os.path.basename(input_file)), aligned_face)
                else:
                    cv2.imwrite(os.path.join(output_path, '%03d_%08d_%03d.jpg' % (input_index, 0, i)), aligned_face)
                detected += 1
            else:
                rejected += 1
    else:
        print('invalid input file:', input_file)

    return detected, rejected

def detect_dir(mtcnn, dirname, output, args):
    input_files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    input_index = 0
    file_count = 0
    total_detected = 0
    total_rejected = 0
    print(dirname)
    for input_file in input_files:
        if os.path.isdir(input_file):
            new_output = os.path.join(output, os.path.basename(input_file))
            if not os.path.exists(new_output):
                os.mkdir(new_output)
            detected, rejected, count = detect_dir(mtcnn, input_file, new_output, args)
        else:
            detected, rejected = detect_file(mtcnn, input_file, input_index, output, args)
            count = 1
        file_count += count
        total_detected += detected
        total_rejected += rejected
        input_index += 1
    return total_detected, total_rejected, file_count

def detect_file(mtcnn, input_file, input_index, output, args):
    video_ext = ['.mkv', '.mp4', '.avi', '.wmv']
    image_ext = ['.jpg', '.jpeg', '.png', '.gif']

    filename, file_ext = os.path.splitext(input_file)
    if file_ext in video_ext:
        return detect_video(mtcnn, input_file, input_index, output, args)
    elif file_ext in image_ext:
        return detect_image(mtcnn, input_file, input_index, output, args)
    else:
        print('Unknown file format:', file_ext[1:])
    return 0, 0

def extract_face_images(args):
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        mtcnn = MTCNN(sess, multiple=args.detect_multiple_faces)

        if os.path.isdir(args.input):
            detected, rejected, count = detect_dir(mtcnn, args.input, args.output, args)
        else:
            detected, rejected = detect_file(mtcnn, args.input, 0, args.output, args)
            count = 1

        print('total files: ', count)
        print('total faces:', detected, 'total rejected:', rejected)
        
def face_detection_benchmark(args):
    print('Face Detection Becnchmark')
    selected_category = [
        '1--',
        '9--',
        '11--',
        '12--',
        '49--'
        ]
    dirs = []
    for x in os.listdir(args.input):
        for cat in selected_category:
            if x.find(cat) == 0:
                dirs.append(os.path.join(args.input, x))
                break
            
    outfile = open(args.output, 'wt')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        mtcnn = MTCNN(sess, multiple=args.detect_multiple_faces)
        mtcnn.margin_rate = 0.0     # no margin
        mtcnn.threshold = [ 0.6, 0.7, 0.1 ]
        for inputpath in dirs:
            inputfiles = [os.path.join(inputpath, x) for x in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, x))]
            for inputfile in inputfiles:
                image = cv2.imread(inputfile)
                
                detected_faces, detected_bb, f_points, face_score = mtcnn.detect(image)
              
                filepath = inputfile[len(args.input)+1:]
                outfile.write(filepath+'\n')
                outfile.write(str(len(detected_bb))+'\n')
                for i in range(len(detected_bb)):
                    bb = detected_bb[i]
                    outfile.write(str(bb[0]) + ' ' + str(bb[1]) + ' ' + str(bb[2]-bb[0]) + ' ' + str(bb[3]-bb[1])+ ' ' + str(face_score[i]) +  '\n')
                
    outfile.close()
            

if __name__ == '__main__':
    print('register.py - denev - face registration and other set of utilities')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to the input video file')
    parser.add_argument('--output', help='path to the output images')
    parser.add_argument('--interval', type=int, default=1000, help='saving interval in milliseconds')
    parser.add_argument('--image_size', type=int, default=160, help='size of output image')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser.add_argument('--detect_multiple_faces', type=int,
                        help='Detect and align multiple faces per image.', default=1)
    parser.add_argument('--preserve_file_name', type=int, default=0)
    parser.add_argument('--benchmark', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    if args.benchmark == 0:
        extract_face_images(args)
    elif args.benchmark == 1:
        face_detection_benchmark(args)
