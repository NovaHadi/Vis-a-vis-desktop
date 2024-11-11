# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 06:55:19 2022

@author: N.H.Lestriandoko
"""

import numpy as np
import cv2
import dlib

import face_replacement  as fpr
import face_replacement_NDTS  as ndts
#import matplotlib.pyplot as plt
import face_morphing as mp

from face_alignment import manual_aligning_68_v3
from imutils import face_utils


class face_swap():
    def __init__(self, input_img1, input_img2, parts='eyebrows-eyes-nose-mouth', method='NDTS', dlib_path = 'D:/dlib-models-master/'):
        #predictor5_path = 'D:/dlib-models-master/shape_predictor_5_face_landmarks.dat'
        predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
        face_rec_model_path = dlib_path +'dlib_face_recognition_resnet_model_v1.dat'
        #modelPath = dlib_path +'mmod_human_face_detector.dat'

        self.input_img1 = input_img1
        self.input_img2 = input_img2
        self.parts = parts
        self.method = method
        
        self.predictor = dlib.shape_predictor(predictor68_path)
        #self.detector_cnn = dlib.cnn_face_detection_model_v1(modelPath)
        self.detector = dlib.get_frontal_face_detector()
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    def swap(self):
        img1 = dlib.load_rgb_image(self.input_img1)
        img2 = dlib.load_rgb_image(self.input_img2)
        
        if self.method == 'NDTS':
            print(f'[INFO] NDTS : {self.parts}')
            output, d_img1, d_img2, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(img1, img2, self.parts, self.detector, self.predictor)

        elif self.method == 'NDT':
            print(f'[INFO] NDT : {self.parts}')
            output, morphed_img, mask_img, delaunay_img1, delaunay_img2, all_points, ori_points = fpr.face_part_replacement(img1, self.detector, self.predictor, img2, 
                               alpha=1, facepart_type=self.parts)

        elif self.method == 'NDS':
            print(f'[INFO] NDS : {self.parts}')
            face1 = self.detector(img1, 2)
            landmarks1 = self.predictor(img1, face1[0])
            shape1 = face_utils.shape_to_np(landmarks1)
            face2 = self.detector(img2, 2)
            landmarks2 = self.predictor(img2, face2[0])
            shape2 = face_utils.shape_to_np(landmarks2)

            output = manual_aligning_68_v3(img1, shape1, shape2)

        else :
            print(f'[INFO] FullFace : {self.parts}')
            face1 = self.detector(img1, 2)
            landmarks1 = self.predictor(img1, face1[0])
            shape1 = face_utils.shape_to_np(landmarks1)
            face2 = self.detector(img2, 2)
            landmarks2 = self.predictor(img2, face2[0])
            shape2 = face_utils.shape_to_np(landmarks2)

            w = 8
            #points1 = shape[0:17] # face border
            points1 = shape1[0:68] # whole face
            points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                             [shape1[18][0],shape1[18][1]-w],
                             [shape1[19][0],shape1[19][1]-w],
                             [shape1[24][0],shape1[24][1]-w],
                             [shape1[25][0],shape1[25][1]-w],
                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                             #[0,0],[0,149],[149,0],[149,149]], axis=0)

            points2 = shape2[0:68] # whole face
            points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                             [shape2[18][0],shape2[18][1]-w],
                             [shape2[19][0],shape2[19][1]-w],
                             [shape2[24][0],shape2[24][1]-w],
                             [shape2[25][0],shape2[25][1]-w],
                             [shape2[26][0],shape2[26][1]-w]], axis=0)
                             #[0,0],[0,149],[149,0],[149,149]], axis=0)

            d_img1 = img1.copy()
            d_img2 = img2.copy()
            mask_img = np.zeros(img2.shape, dtype = np.float32)
            
            # ================ iNDTS whole face ===============            
            #morphed_landmark, del_img1, del_img2 = manual_aligning_68_v3(meanFace, points2, points1, img)
            
            morphed_landmark = manual_aligning_68_v3(img2, points2, points1)
            #dlib.save_image(morphed_landmark, output_filename)

            output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
            #dlib.save_image(output_replacement, output_filename)
            
            center, src_mask = getMaskCenter(morphed_landmark, mask)
            output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
            #dlib.save_image(output, output_filename)            
        return output

def getMaskCenter(img1, mask):
    src_mask = np.zeros(img1.shape, img1.dtype)
    src_mask[mask>0] = 255
    poly = np.argwhere(mask[:,:,2]>0)
    r = cv2.boundingRect(np.float32([poly]))    
    center = (r[1]+int(r[3]/2)), (r[0]+int(r[2]/2))
    return center, src_mask