# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:16:22 2022

@author: N.H. Lestriandoko

"""
import os
import dlib
import numpy as np
import cv2
import face_replacement  as fpr
import face_replacement_NDTS  as ndts
import face_morphing as mp
import general as gn

from face_alignment import manual_aligning_68_v3
from imutils import face_utils

class folder_swap():
    def __init__(self, input_folder, output_folder, input_avgFace, folder_type=1, parts='eyebrows-eyes-nose-mouth', method='NDTS', bInverse=False, dlib_path = 'D:/dlib-models-master/'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_avgFace = input_avgFace
        self.folder_type = folder_type
        self.parts = parts
        self.method = method
        self.bInverse = bInverse
        self.dlib_path = dlib_path

        predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor68_path)
        self.detector = dlib.get_frontal_face_detector()
    
    def swap(self):
        if self.folder_type == 1:
            swap_folder_1(self.input_folder, self.output_folder, self.input_avgFace, self.method, self.bInverse, self.parts, self.detector, self.predictor )
        else:
            swap_folder_2(self.input_folder, self.output_folder, self.input_avgFace, self.method, self.bInverse, self.parts, self.detector, self.predictor )
            
def check_multiple_faces(face_rects):
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
    return face_rect

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

def getMaskCenter(img1, mask):
    src_mask = np.zeros(img1.shape, img1.dtype)
    src_mask[mask>0] = 255
    poly = np.argwhere(mask[:,:,2]>0)
    r = cv2.boundingRect(np.float32([poly]))    
    center = (r[1]+int(r[3]/2)), (r[0]+int(r[2]/2))
    return center, src_mask 
        
def swap_folder_1(input_dir,output_dir,input_avgFace, method, bInverse, parts, detector, predictor):
    nrof_images = 0
    nrof_successfully_aligned = 0
    
    img2 = dlib.load_rgb_image(input_avgFace)
    face2 = detector(img2, 2)
    landmarks2 = predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    path_input = os.path.expanduser(input_dir)    
    for filename in os.listdir(path_input):
        split_filename = os.path.splitext(os.path.split(filename)[1])[0]
        output_filename = os.path.join(output_dir, split_filename+'.png')
        nrof_images += 1
        
        img1 = dlib.load_rgb_image(os.path.join(input_dir, filename))
        
        face_rects = detector(img1, 2)
        if len(face_rects)>0:
            face_rect = check_multiple_faces(face_rects)                    
                
            # extract the landmarks
            landmarks1 = predictor(img1, face_rect)
            shape1 = face_utils.shape_to_np(landmarks1)
            
            if method == 'NDTS':
                if bInverse:
                    output, d_img2, d_img1, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(img2, img1, parts, detector, predictor)
                else:
                    output, d_img1, d_img2, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(img1, img2, parts, detector, predictor)
                    
            elif method == 'NDT':
                if bInverse:
                    output, morphed_img, mask_img, delaunay_img2, delaunay_img1, all_points, ori_points = fpr.face_part_replacement(img2, detector, predictor, img1, 
                                   alpha=1, facepart_type=parts)
                else:
                    output, morphed_img, mask_img, delaunay_img1, delaunay_img2, all_points, ori_points = fpr.face_part_replacement(img1, detector, predictor, img2, 
                                   alpha=1, facepart_type=parts)                    
    
            elif method == 'NDS':
                if bInverse:
                    output = manual_aligning_68_v3(img2, shape2, shape1)
                else:
                    output = manual_aligning_68_v3(img1, shape1, shape2)                    
                    
            elif method == 'full-face':
    
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
                
                if bInverse:
                    morphed_landmark = manual_aligning_68_v3(img2, points2, points1)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                else:
                    morphed_landmark = manual_aligning_68_v3(img1, points1, points2)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

            elif method == 'half-face':
    
                w = 8
                #points1 = shape[0:17] # face border
                points1_ff = shape1[0:68]
                points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                
                points1 = shape1[15:48] # half face
                add_points1 = shape1[0:2]
                points1 = np.append(points1,add_points1,axis=0)
                points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[36][0],shape1[2][1]],
                                 [shape1[45][0],shape1[14][1]],
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                points2_ff = shape2[0:68]
                points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)

                points2 = shape2[15:48] # half face
                add_points2 = shape2[0:2]
                points2 = np.append(points2,add_points2,axis=0)
                points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[36][0],shape2[2][1]],
                                 [shape2[45][0],shape2[14][1]],
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                d_img1 = img1.copy()
                d_img2 = img2.copy()
                mask_img = np.zeros(img2.shape, dtype = np.float32)
                
                if bInverse:
                    morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                else:
                    morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

            elif method == 'half-face-cheek':
    
                #w = 8      # for 150 x 150 pixels 
                w = 8 * 3  # for 500 x 500 pixels
                #points1 = shape[0:17] # face border
                points1_ff = shape1[0:68]
                points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                
                #points1 = shape1[15:48] # half face without mouth
                points1 = shape1[15:60] # half face with mouth
                add_points1 = shape1[0:2]
                points1 = np.append(points1,add_points1,axis=0)
                points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[36][0],shape1[2][1]],
                                 [shape1[45][0],shape1[14][1]],
                                 [shape1[41][0],shape1[48][1]],
                                 [shape1[46][0],shape1[54][1]],
                                 [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                 [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                 [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                points2_ff = shape2[0:68]
                points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)

                #points2 = shape2[15:48] # half face without mouth
                points2 = shape2[15:60] # half face with mouth
                add_points2 = shape2[0:2]
                points2 = np.append(points2,add_points2,axis=0)
                points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[36][0],shape2[2][1]],
                                 [shape2[45][0],shape2[14][1]],
                                 [shape2[41][0],shape2[48][1]],
                                 [shape2[46][0],shape2[54][1]],
                                 [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                 [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                 [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                d_img1 = img1.copy()
                d_img2 = img2.copy()
                mask_img = np.zeros(img2.shape, dtype = np.float32)
                
                if bInverse:
                    morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                else:
                    morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

            elif method == 'face-without-hair-beard':
    
                #w = 8      # for 150 x 150 pixels 
                w = 8 * 3  # for 500 x 500 pixels
                #points1 = shape[0:17] # face border
                points1_ff = shape1[0:68]
                points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                
                #points1 = shape1[15:48] # half face without mouth
                points1 = shape1[15:60] # half face with mouth
                add_points1 = shape1[0:2]
                points1 = np.append(points1,add_points1,axis=0)
                points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[36][0],shape1[2][1]],
                                 [shape1[45][0],shape1[14][1]],
                                 [shape1[41][0],shape1[48][1]],
                                 [shape1[46][0],shape1[54][1]],
                                 [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                 [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                 [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                points2_ff = shape2[0:68]
                points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)

                #points2 = shape2[15:48] # half face without mouth
                points2 = shape2[15:60] # half face with mouth
                add_points2 = shape2[0:2]
                points2 = np.append(points2,add_points2,axis=0)
                points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[36][0],shape2[2][1]],
                                 [shape2[45][0],shape2[14][1]],
                                 [shape2[41][0],shape2[48][1]],
                                 [shape2[46][0],shape2[54][1]],
                                 [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                 [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                 [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)
                                 #[0,0],[0,149],[149,0],[149,149]], axis=0)
    
                d_img1 = img1.copy()
                d_img2 = img2.copy()
                mask_img = np.zeros(img2.shape, dtype = np.float32)
                
                if bInverse:
                    morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                else:
                    morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

            elif method == 'half-face-cheek-v2':
                # updated method of the half-face-cheek
                # remove the remaining hair on the edge of the face.
                
                #w = 8      # for 150 x 150 pixels 
                w = 8 * 3  # for 500 x 500 pixels
                left_edge1 = 0
                right_edge1 = 0
                left_edge2 = 0
                right_edge2 = 0
                if shape1[16][0]-shape1[45][0]>30:
                    left_edge1 = int((shape1[16][0]-shape1[45][0])/2)
                    left_edge2 = int((shape2[16][0]-shape2[45][0])/2)
                if shape1[36][0]-shape1[0][0]>30:
                    right_edge1 = int((shape1[36][0]-shape1[0][0])/2)
                    right_edge2 = int((shape2[36][0]-shape2[0][0])/2)
                #points1 = shape[0:17] # face border
                points1_ff = shape1[0:68]
                points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w]], axis=0)
                
                points1 = shape1[17:60] # half face with mouth
                points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                 [shape1[36][0],shape1[2][1]],
                                 [shape1[45][0],shape1[14][1]],
                                 [shape1[41][0],shape1[48][1]],
                                 [shape1[46][0],shape1[54][1]],
                                 [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                 [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                 [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                 [shape1[18][0],shape1[18][1]-w],
                                 [shape1[19][0],shape1[19][1]-w],
                                 [shape1[24][0],shape1[24][1]-w],
                                 [shape1[25][0],shape1[25][1]-w],
                                 [shape1[26][0],shape1[26][1]-w],
                                 [shape1[0][0]+right_edge1,shape1[0][1]],
                                 [shape1[1][0]+right_edge1,shape1[1][1]],
                                 [shape1[15][0]-left_edge1,shape1[15][1]],
                                 [shape1[16][0]-left_edge1,shape1[16][1]]], axis=0)
    
                points2_ff = shape2[0:68]
                points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w]], axis=0)

                points2 = shape2[17:60] # half face with mouth
                points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                 [shape2[36][0],shape2[2][1]],
                                 [shape2[45][0],shape2[14][1]],
                                 [shape2[41][0],shape2[48][1]],
                                 [shape2[46][0],shape2[54][1]],
                                 [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                 [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                 [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                 [shape2[18][0],shape2[18][1]-w],
                                 [shape2[19][0],shape2[19][1]-w],
                                 [shape2[24][0],shape2[24][1]-w],
                                 [shape2[25][0],shape2[25][1]-w],
                                 [shape2[26][0],shape2[26][1]-w],
                                 [shape2[0][0]+right_edge2,shape2[0][1]],
                                 [shape2[1][0]+right_edge2,shape2[1][1]],
                                 [shape2[15][0]-left_edge2,shape2[15][1]],
                                 [shape2[16][0]-left_edge2,shape2[16][1]]], axis=0)
    
                d_img1 = img1.copy()
                d_img2 = img2.copy()
                mask_img = np.zeros(img2.shape, dtype = np.float32)
                
                if bInverse:
                    morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                else:
                    morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                    
                    center, src_mask = getMaskCenter(morphed_landmark, mask)
                    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

                    
            dlib.save_image(output, output_filename)
            nrof_successfully_aligned += 1
                    
    print('Total number of images: %d' % nrof_images)
    print('Number of successfully swap images: %d' % nrof_successfully_aligned)

def swap_folder_2(input_dir,output_dir,input_avgFace, method, bInverse, parts, detector, predictor):
    nrof_images = 0
    nrof_successfully_aligned = 0
    dataset = gn.get_dataset(input_dir)

    img2 = dlib.load_rgb_image(input_avgFace)
    face2 = detector(img2, 2)
    landmarks2 = predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
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
                    img1 = dlib.load_rgb_image(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    face_rects = detector(img1, 2)
                    if len(face_rects)>0:
                                            
                        face_rect = check_multiple_faces(face_rects)                    
                            
                        # extract the landmarks
                        landmarks1 = predictor(img1, face_rect)
                        shape1 = face_utils.shape_to_np(landmarks1)
                        
                        if method == 'NDTS':
                            if bInverse:
                                output, d_img2, d_img1, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(img2, img1, parts, detector, predictor)
                            else:
                                output, d_img1, d_img2, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(img1, img2, parts, detector, predictor)
                
                        elif method == 'NDT':
                            if bInverse:
                                output, morphed_img, mask_img, delaunay_img2, delaunay_img1, all_points, ori_points = fpr.face_part_replacement(img2, detector, predictor, img1, 
                                               alpha=1, facepart_type=parts)
                            else:
                                output, morphed_img, mask_img, delaunay_img1, delaunay_img2, all_points, ori_points = fpr.face_part_replacement(img1, detector, predictor, img2, 
                                               alpha=1, facepart_type=parts)                    
                
                        elif method == 'NDS':
                            if bInverse:
                                output = manual_aligning_68_v3(img2, shape2, shape1)
                            else:
                                output = manual_aligning_68_v3(img1, shape1, shape2)                    
                        
                        elif method == 'full-face':
                
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
                            
                            if bInverse:
                                morphed_landmark = manual_aligning_68_v3(img2, points2, points1)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                            else:
                                morphed_landmark = manual_aligning_68_v3(img1, points1, points2)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

                        elif method == 'half-face':
                
                            w = 8
                            #points1 = shape[0:17] # face border
                            points1_ff = shape1[0:68]
                            points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                            
                            points1 = shape1[15:48] # half face
                            add_points1 = shape1[0:2]
                            points1 = np.append(points1,add_points1,axis=0)
                            points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w], # above right eyebrows - point 18
                                             [shape1[36][0],shape1[2][1]],     # below right eye corner - point 37x,3y
                                             [shape1[45][0],shape1[14][1]],    # below left eye corner - point 46x,15y
                                             [shape1[18][0],shape1[18][1]-w],  # above right eyebrows - point 19
                                             [shape1[19][0],shape1[19][1]-w],  # above right eyebrows - point 20
                                             [shape1[24][0],shape1[24][1]-w],  # above left eyebrows - point 25
                                             [shape1[25][0],shape1[25][1]-w],  # above left eyebrows - point 26
                                             [shape1[26][0],shape1[26][1]-w]], axis=0) # above left eyebrows - point 27
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            points2_ff = shape2[0:68]
                            points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)

                            points2 = shape2[15:48] # half face
                            add_points2 = shape2[0:2]
                            points2 = np.append(points2,add_points2,axis=0)
                            points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[36][0],shape2[2][1]],     # below right eye corner
                                             [shape2[45][0],shape2[14][1]],    # below left eye corner
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            d_img1 = img1.copy()
                            d_img2 = img2.copy()
                            mask_img = np.zeros(img2.shape, dtype = np.float32)
                            
                            if bInverse:
                                morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                            else:
                                morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                                                        
                        elif method == 'half-face-cheek':
                
                            #w = 8      # for 150 x 150 pixels 
                            w = 8 * 3  # for 500 x 500 pixels
                            #points1 = shape[0:17] # face border
                            points1_ff = shape1[0:68]
                            points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                            
                            #points1 = shape1[15:48] # half face without mouth
                            points1 = shape1[15:60] # half face with mouth
                            add_points1 = shape1[0:2]
                            points1 = np.append(points1,add_points1,axis=0)
                            points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[36][0],shape1[2][1]],
                                             [shape1[45][0],shape1[14][1]],
                                             [shape1[41][0],shape1[48][1]],
                                             [shape1[46][0],shape1[54][1]],
                                             [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                             [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                             [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            points2_ff = shape2[0:68]
                            points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)

                            #points2 = shape2[15:48] # half face without mouth
                            points2 = shape2[15:60] # half face with mouth
                            add_points2 = shape2[0:2]
                            points2 = np.append(points2,add_points2,axis=0)
                            points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[36][0],shape2[2][1]],
                                             [shape2[45][0],shape2[14][1]],
                                             [shape2[41][0],shape2[48][1]],
                                             [shape2[46][0],shape2[54][1]],
                                             [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                             [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                             [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            d_img1 = img1.copy()
                            d_img2 = img2.copy()
                            mask_img = np.zeros(img2.shape, dtype = np.float32)
                            
                            if bInverse:
                                morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                            else:
                                morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

                        elif method == 'face-without-hair-beard':
                
                            #w = 8      # for 150 x 150 pixels 
                            w = 8 * 3  # for 500 x 500 pixels
                            #points1 = shape[0:17] # face border
                            points1_ff = shape1[0:68]
                            points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                            
                            #points1 = shape1[15:48] # half face without mouth
                            points1 = shape1[15:60] # half face with mouth
                            add_points1 = shape1[0:2]
                            points1 = np.append(points1,add_points1,axis=0)
                            points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[36][0],shape1[2][1]],
                                             [shape1[45][0],shape1[14][1]],
                                             [shape1[41][0],shape1[48][1]],
                                             [shape1[46][0],shape1[54][1]],
                                             [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                             [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                             [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            points2_ff = shape2[0:68]
                            points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)

                            #points2 = shape2[15:48] # half face without mouth
                            points2 = shape2[15:60] # half face with mouth
                            add_points2 = shape2[0:2]
                            points2 = np.append(points2,add_points2,axis=0)
                            points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[36][0],shape2[2][1]],
                                             [shape2[45][0],shape2[14][1]],
                                             [shape2[41][0],shape2[48][1]],
                                             [shape2[46][0],shape2[54][1]],
                                             [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                             [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                             [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)
                                             #[0,0],[0,149],[149,0],[149,149]], axis=0)
                
                            d_img1 = img1.copy()
                            d_img2 = img2.copy()
                            mask_img = np.zeros(img2.shape, dtype = np.float32)
                            
                            if bInverse:
                                morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                            else:
                                morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

                        elif method == 'half-face-cheek-v2':
                            # updated method of the half-face-cheek
                            # remove the remaining hair on the edge of the face.
                            
                            #w = 8      # for 150 x 150 pixels 
                            w = 8 * 3  # for 500 x 500 pixels
                            left_edge1 = 0
                            right_edge1 = 0
                            left_edge2 = 0
                            right_edge2 = 0
                            if shape1[16][0]-shape1[45][0]>30:
                                left_edge1 = int((shape1[16][0]-shape1[45][0])/2)
                                left_edge2 = int((shape2[16][0]-shape2[45][0])/2)
                            if shape1[36][0]-shape1[0][0]>30:
                                right_edge1 = int((shape1[36][0]-shape1[0][0])/2)
                                right_edge2 = int((shape2[36][0]-shape2[0][0])/2)
                            #points1 = shape[0:17] # face border
                            points1_ff = shape1[0:68]
                            points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w]], axis=0)
                            
                            points1 = shape1[17:60] # half face with mouth
                            points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                                             [shape1[36][0],shape1[2][1]],
                                             [shape1[45][0],shape1[14][1]],
                                             [shape1[41][0],shape1[48][1]],
                                             [shape1[46][0],shape1[54][1]],
                                             [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                                             [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                                             [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                                             [shape1[18][0],shape1[18][1]-w],
                                             [shape1[19][0],shape1[19][1]-w],
                                             [shape1[24][0],shape1[24][1]-w],
                                             [shape1[25][0],shape1[25][1]-w],
                                             [shape1[26][0],shape1[26][1]-w],
                                             [shape1[0][0]+right_edge1,shape1[0][1]],
                                             [shape1[1][0]+right_edge1,shape1[1][1]],
                                             [shape1[15][0]-left_edge1,shape1[15][1]],
                                             [shape1[16][0]-left_edge1,shape1[16][1]]], axis=0)
                
                            points2_ff = shape2[0:68]
                            points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w]], axis=0)

                            points2 = shape2[17:60] # half face with mouth
                            points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                                             [shape2[36][0],shape2[2][1]],
                                             [shape2[45][0],shape2[14][1]],
                                             [shape2[41][0],shape2[48][1]],
                                             [shape2[46][0],shape2[54][1]],
                                             [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                                             [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                                             [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                                             [shape2[18][0],shape2[18][1]-w],
                                             [shape2[19][0],shape2[19][1]-w],
                                             [shape2[24][0],shape2[24][1]-w],
                                             [shape2[25][0],shape2[25][1]-w],
                                             [shape2[26][0],shape2[26][1]-w],
                                             [shape2[0][0]+right_edge2,shape2[0][1]],
                                             [shape2[1][0]+right_edge2,shape2[1][1]],
                                             [shape2[15][0]-left_edge2,shape2[15][1]],
                                             [shape2[16][0]-left_edge2,shape2[16][1]]], axis=0)
                
                            d_img1 = img1.copy()
                            d_img2 = img2.copy()
                            mask_img = np.zeros(img2.shape, dtype = np.float32)
                            
                            if bInverse:
                                morphed_landmark = manual_aligning_68_v3(img2, points2_ff, points1_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img1, d_img2, d_img1, mask_img, points2, points1, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
                            else:
                                morphed_landmark = manual_aligning_68_v3(img1, points1_ff, points2_ff)    
                                output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
                                
                                center, src_mask = getMaskCenter(morphed_landmark, mask)
                                output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)


                        dlib.save_image(output, output_filename)

                        nrof_successfully_aligned += 1

    print('Total number of images: %d' % nrof_images)
    print('Number of successfully swap images: %d' % nrof_successfully_aligned)


