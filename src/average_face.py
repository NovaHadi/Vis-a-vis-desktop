# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 03:32:45 2022

@author: N.H.Lestriandoko
"""
import numpy as np
import os
import dlib
import general as gn
from imutils import face_utils
from face_alignment import manual_aligning_68_v3

#import matplotlib.pyplot as plt

class average_face():
    def __init__(self, input_dir, output_file, folder_type=1, dlib_path = 'D:/dlib-models-master/', mean_points = 'D:/Datasets/RomanEmperors2/Dataset_07-06-2022/AverageFace_Calculation/2_meanPoints/meanPoints.npz'):
        predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
        #face_rec_model_path = dlib_path +'dlib_face_recognition_resnet_model_v1.dat'
        self.predictor = dlib.shape_predictor(predictor68_path)
        self.detector = dlib.get_frontal_face_detector()

        self.input_dir = input_dir
        self.output_file = output_file
        self.folder_type = folder_type
        self.mean_points = mean_points

    def average_faces(self):
        if self.folder_type == 1:
            avgFace, nrof_images = calculate_average_face_folder_1(self.input_dir, self.output_file)
        else:
            avgFace, nrof_images = calculate_average_face_folder_2(self.input_dir, self.output_file)

        #plt.figure()
        #plt.imshow(np.uint8(avgFace))
        #plt.axis('off')
        #plt.title('Average Face')

        #filename = os.path.splitext(os.path.split(self.output_file)[1])[0]
        output_class_dir = os.path.splitext(self.output_file)[0]
        output_filename = output_class_dir +'.png'
        dlib.save_image(np.uint8(avgFace), output_filename)


    def average_points(self):
        if self.folder_type == 1:
            avgFace, nrof_images = calculate_average_points_folder_1(self.input_dir, self.output_file, self.detector, self.predictor)
        else:
            avgFace, nrof_images = calculate_average_points_folder_2(self.input_dir, self.output_file, self.detector, self.predictor)

    def manual_align(self):
        mean_LM_filepath = self.mean_points
        dataLM = np.load(mean_LM_filepath)
        mean_points = dataLM['mean_points']
        if self.folder_type == 1:
            realign_V3_folder_1(self.input_dir, mean_points, self.output_file, self.detector, self.predictor)
        else:
            realign_V3_folder_2(self.input_dir, mean_points, self.output_file, self.detector, self.predictor)

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
            
def calculate_average_face_folder_1(input_dir, output_file):
    nrof_images = 0
    all_img = []
    
    path_input = os.path.expanduser(input_dir)
    
    for filename in os.listdir(path_input):
        nrof_images += 1
        img = dlib.load_rgb_image(os.path.join(input_dir, filename))
        all_img.append(img)
        
    images = np.stack(all_img)
    mean_face = np.average(images, axis=0)
    np.savez(output_file, mean_face=mean_face)
    print('Total number of images: %d' % nrof_images)
    return mean_face, nrof_images

def calculate_average_face_folder_2(input_dir, output_file):
    nrof_images = 0
    all_img = []
    
    dataset = gn.get_dataset(input_dir)
        
    for cls in dataset:
        for image_path in cls.image_paths:
            nrof_images += 1
            img = dlib.load_rgb_image(image_path)
            all_img.append(img)

    images = np.stack(all_img)
    mean_face = np.average(images, axis=0)
    np.savez(output_file, mean_face=mean_face)
    print('Total number of images: %d' % nrof_images)
    return mean_face, nrof_images

def calculate_average_points_folder_1(input_dir, output_file, detector, predictor):
    nrof_images = 0
    points = []
    
    path_input = os.path.expanduser(input_dir)
    
    for filename in os.listdir(path_input):
        nrof_images += 1
        img = dlib.load_rgb_image(os.path.join(input_dir, filename))
        face = detector(img, 0)
        if len(face)>0:
            # extract the landmarks
            landmarks = predictor(img, face[0])
            # to np.array
            shape = face_utils.shape_to_np(landmarks)
            points.append(shape)
        else:
            print(filename)
    points_images = np.stack(points)

    mean_points = np.average(points_images, axis=0)
    np.savez(output_file, mean_points=mean_points)
    print('Total number of images: %d' % nrof_images)
    return mean_points, nrof_images

def calculate_average_points_folder_2(input_dir, output_file, detector, predictor):
    nrof_images = 0
    points = []
    
    dataset = gn.get_dataset(input_dir)
        
    for cls in dataset:
        for image_path in cls.image_paths:
            nrof_images += 1
            img = dlib.load_rgb_image(image_path)
            face = detector(img, 0)
            if len(face)>0:
                # extract the landmarks
                landmarks = predictor(img, face[0])
                # to np.array
                shape = face_utils.shape_to_np(landmarks)
                points.append(shape)
            else:
                print(image_path)
    points_images = np.stack(points)

    mean_points = np.average(points_images, axis=0)
    np.savez(output_file, mean_points=mean_points)
    print('Total number of images: %d' % nrof_images)
    return mean_points, nrof_images

def realign_V3_folder_1(input_dir, mean_points, output_dir, detector, predictor):
    nrof_failed = 0
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    path_input = os.path.expanduser(input_dir)
    
    for filename in os.listdir(path_input):

        split_filename = os.path.splitext(os.path.split(filename)[1])[0]
        output_filename = os.path.join(output_dir, split_filename+'.png')

        img = dlib.load_rgb_image(os.path.join(input_dir, filename))

        face = detector(img, 0)
        if len(face)>0:
            nFaces = len(face)            
            if nFaces > 1:
                x1, y1, w1, h1 = rect_to_bb(face[0])
                x2, y2, w2, h2 = rect_to_bb(face[1])
                if w1>w2:
                    face_rect = face[0]
                else:
                    face_rect = face[1]
            else:
                face_rect = face[0]
            # extract the landmarks
            landmarks = predictor(img, face_rect)
            # to np.array
            shape = face_utils.shape_to_np(landmarks)
            img_align = manual_aligning_68_v3(img, shape, mean_points)
            dlib.save_image(img_align, output_filename)
        else:
            nrof_failed += 1
            print('[DETECTION FAILED] ' + input_dir + filename)
        
    print('Number of failed detection: %d' % nrof_failed)

def realign_V3_folder_2(input_dir, mean_points, output_dir, detector, predictor):
    nrof_images = 0
    nrof_failed = 0
    dataset = gn.get_dataset(input_dir)
    
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        for image_path in cls.image_paths:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print(image_path)

            nrof_images += 1

            if not os.path.exists(output_filename):
                try:
                    img = dlib.load_rgb_image(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    face = detector(img, 0)
                    if len(face)>0:
                        nFaces = len(face)            
                        if nFaces > 1:
                            x1, y1, w1, h1 = rect_to_bb(face[0])
                            x2, y2, w2, h2 = rect_to_bb(face[1])
                            if w1>w2:
                                face_rect = face[0]
                            else:
                                face_rect = face[1]
                        else:
                            face_rect = face[0]
                        # extract the landmarks
                        landmarks = predictor(img, face_rect)
                        # to np.array
                        shape = face_utils.shape_to_np(landmarks)
                        img_align = manual_aligning_68_v3(img, shape, mean_points)
                        dlib.save_image(img_align, output_filename)
                    else:
                        nrof_failed += 1
                        print('[DETECTION FAILED] ' + input_dir + filename)


    print('Total number of images: %d' % nrof_images)
    print('Number of failed detection: %d' % nrof_failed)

