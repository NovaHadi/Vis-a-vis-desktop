# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 04:32:53 2022

@author: N.H. Lestriandoko
"""
import numpy as np
import cv2
import dlib
import os

#import matplotlib.pyplot as plt
import general as gn

"""
folder_type:
    0 : filename path
    1 : folder 1 level
    2 : folder 2 levels
"""
class face_detection():
    def __init__(self, input_dir, output_dir, folder_type=0, padding=0.25, size=150, dlib_path = 'D:/dlib-models-master/'):
        #self.predictor5_path = dlib_path +'shape_predictor_5_face_landmarks.dat'
        self.predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
        self.face_rec_model_path = dlib_path +'dlib_face_recognition_resnet_model_v1.dat'
        #self.modelPath = dlib_path +'mmod_human_face_detector.dat'

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.folder_type = folder_type
        self.padding = padding
        self.size = size
        
        self.predictor = dlib.shape_predictor(self.predictor68_path)
        #self.detector_cnn = dlib.cnn_face_detection_model_v1(self.modelPath)
        self.detector = dlib.get_frontal_face_detector()
        #self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)
        
    def detect_faces(self):
        if self.folder_type == 0:
            print(f'[INFO] padding : {self.padding}')
            alignment_singlefile_DLIB_folder_0(self.input_dir, self.output_dir, self.detector, self.predictor, self.padding, self.size)
        elif self.folder_type == 1:
            alignment_dataset_DLIB_folder_1(self.input_dir, self.output_dir, self.detector, self.predictor, self.padding, self.size)           
        else:
            alignment_dataset_DLIB_folder_2(self.input_dir, self.output_dir, self.detector, self.predictor, self.padding, self.size)           
            #print(f'[INFO] folder type : {self.folder_type}')



def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

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

def crop_to_bbox(img, bbox):
    shape_img = img.shape[:2]
    bbox = np.clip(bbox, 0., 1.)
    bbox = np.reshape(bbox, [-1, 2]) * shape_img
    bbox = bbox.flatten().astype(np.int32)
    img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] # crop
    return img_cropped

def alignment_singlefile_DLIB_folder_0(input_dir, output_dir, detector, predictor, padding, size):
    
    print(f'input_path: {input_dir}' )
    print(f'output_path: {output_dir}' )
    
    img = dlib.load_rgb_image(input_dir)
    img_crop = img
    #plt.figure()
    #plt.imshow(img)

    #face_rects = detector(img, 2)
    #if len(face_rects)<1:
    if img.shape[0]<img.shape[1]:
        img = rotate_bound(img, angle=90)
        face_rects = detector(img, 2)
        if len(face_rects)<1:
            img = rotate_bound(img, angle=270)
            face_rects = detector(img, 2)
    else :
        face_rects = detector(img, 2)
        if len(face_rects)<1:
            img = rotate_bound(img, angle=180)
            face_rects = detector(img, 2)

    if len(face_rects)<1:   
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_rects = detector(gray, 0)
        if len(face_rects)<1:   
            print('[DETECTION FAILED] ' + input_dir)
    """
    if len(face_rects)<1:   
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_rects = detector(gray, 0)
        if len(face_rects)<1:   
            print('[DETECTION FAILED] ' + input_dir)
    """
    if len(face_rects)>0:
                            
        nFaces = len(face_rects)            
        if nFaces > 1:
            x1, y1, w1, h1 = rect_to_bb(face_rects[0])
            x2, y2, w2, h2 = rect_to_bb(face_rects[1])
            if w1>w2:
                face_rect = face_rects[0]
            else:
                face_rect = face_rects[1]
        else:
            face_rect = face_rects[0]
            
        # extract the landmarks
        landmarks = predictor(img, face_rect)
        # align and crop the face         
        img_crop = dlib.get_face_chip(img, landmarks, size=size, padding=padding)
        #plt.figure()
        #plt.imshow(img_crop)
        dlib.save_image(img_crop, output_dir)
        print('[FACE DETECTED]')
    

def alignment_dataset_DLIB_folder_1(input_dir, output_dir, detector, predictor, padding, size):
    nrof_failed = 0
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    path_input = os.path.expanduser(input_dir)
    
    for filename in os.listdir(path_input):
        split_filename = os.path.splitext(os.path.split(filename)[1])[0]
        output_filename = os.path.join(output_dir, split_filename+'.png')
        
        img = dlib.load_rgb_image(os.path.join(input_dir, filename))
        #face_rects = detector(img, 2)
        #if len(face_rects)<1:
        if img.shape[0]<img.shape[1]:
            img = rotate_bound(img, angle=90)
            face_rects = detector(img, 2)
            if len(face_rects)<1:
                img = rotate_bound(img, angle=270)
                face_rects = detector(img, 2)
        else :
            face_rects = detector(img, 2)
            if len(face_rects)<1:
                img = rotate_bound(img, angle=180)
                face_rects = detector(img, 2)

        if len(face_rects)<1:   
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            face_rects = detector(gray, 0)
            if len(face_rects)<1:   
                nrof_failed += 1
                print('[DETECTION FAILED] ' + input_dir+'/'+ filename)
        """
        if len(face_rects)<1:   
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            face_rects = detector(gray, 0)
            if len(face_rects)<1:   
                nrof_failed += 1
                print('[DETECTION FAILED] ' + input_dir + filename)
        """
        if len(face_rects)>0:
                                
            nFaces = len(face_rects)            
            if nFaces > 1:
                x1, y1, w1, h1 = rect_to_bb(face_rects[0])
                x2, y2, w2, h2 = rect_to_bb(face_rects[1])
                if w1>w2:
                    face_rect = face_rects[0]
                else:
                    face_rect = face_rects[1]
            else:
                face_rect = face_rects[0]
                
            # extract the landmarks
            landmarks = predictor(img, face_rect)
            # align and crop the face         
            img_crop = dlib.get_face_chip(img, landmarks, size=size, padding=padding)
            #img_crop = dlib.get_face_chip(img, landmarks)
            dlib.save_image(img_crop, output_filename)
            
            #print('[success]' + output_dir + output_filename)
        
    print('Number of failed detection: %d' % nrof_failed)

def alignment_dataset_DLIB_folder_2(input_dir, output_dir, detector, predictor, padding, size):
    nrof_images = 0
    nrof_failed = 0
    nrof_successfully_aligned = 0
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
                    #face_rects = detector(img, 2)
                    #if len(face_rects)<1:
                    if img.shape[0]<img.shape[1]:
                        img = rotate_bound(img, angle=90)
                        face_rects = detector(img, 2)
                        if len(face_rects)<1:
                            img = rotate_bound(img, angle=270)
                            face_rects = detector(img, 2)
                    else :
                        face_rects = detector(img, 2)
                        if len(face_rects)<1:
                            img = rotate_bound(img, angle=180)
                            face_rects = detector(img, 2)
            
                    if len(face_rects)<1:   
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        face_rects = detector(gray, 0)
                        if len(face_rects)<1:   
                            nrof_failed += 1
                            print('[DETECTION FAILED] ' + input_dir +'/'+ filename)
                    """
                    if len(face_rects)<1:   
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        face_rects = detector(gray, 0)
                        if len(face_rects)<1:   
                            nrof_failed += 1
                            print('[DETECTION FAILED] ' + input_dir + filename)
                    """
                    if len(face_rects)>0:
                                            
                        nFaces = len(face_rects)            
                        if nFaces > 1:
                            x1, y1, w1, h1 = rect_to_bb(face_rects[0])
                            x2, y2, w2, h2 = rect_to_bb(face_rects[1])
                            if w1>w2:
                                face_rect = face_rects[0]
                            else:
                                face_rect = face_rects[1]
                        else:
                            face_rect = face_rects[0]
                            
                        # extract the landmarks
                        landmarks = predictor(img, face_rect)
                        # align and crop the face         
                        img_crop = dlib.get_face_chip(img, landmarks, size=size, padding=padding)
                        #img_crop = dlib.get_face_chip(img, landmarks)
                        dlib.save_image(img_crop, output_filename)

                        nrof_successfully_aligned += 1

    print('Total number of images: %d' % nrof_images)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    print('Number of failed detection: %d' % nrof_failed)

