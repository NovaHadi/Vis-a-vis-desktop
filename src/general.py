# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:04:50 2020

@author: N.H.Lestriandoko
"""

import numpy as np
import os
import math
import cv2
import skimage.filters as filters

from numpy import dot
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter
from imutils import face_utils
from skimage import transform as skitrans

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths)) 
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
    
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def cos_sim(a,b):
    cossim = dot(a, b)/(norm(a)*norm(b))
    return cossim

def computeDETdata_origin(Maps, distance, ThresholdRange):
    GAR = np.zeros(len(ThresholdRange))
    FRR = np.zeros(len(ThresholdRange))
    FAR = np.zeros(len(ThresholdRange))
    
    
    Ind = 0;
    for Threshold in ThresholdRange:    
          
        #TP = Maps * (distance <= Threshold) * (distance>0) #Eucidean distance case --> accept if distance <= Threshold
        TP = Maps * (distance <= Threshold) #Eucidean distance case --> accept if distance <= Threshold
        T  = Maps 
        TPR = TP.sum() / T.sum()
        GAR[Ind] = TPR
          
        FR = Maps * (distance > Threshold) #Eucidean distance case --> reject if distance > Threshold 
        M  = Maps 
        FRRi = FR.sum() / M.sum()
        FRR[Ind] = FRRi
          
        FP = (1 - Maps) * (distance <= Threshold) #Eucidean distance case --> accept if distance <= Threshold 
        N  = (1 - Maps);
        FPR = FP.sum() / N.sum()
        FAR[Ind] = FPR;

        Ind = Ind + 1 
    return GAR,FAR,FRR    

def computeDETdata_euc(Maps, distance, ThresholdRange, diag_maps):
    GAR = np.zeros(len(ThresholdRange))
    FRR = np.zeros(len(ThresholdRange))
    FAR = np.zeros(len(ThresholdRange))
    
    
    Ind = 0;
    for Threshold in ThresholdRange:    
          
        #TP = Maps * (distance <= Threshold) * (distance>0) #Eucidean distance case --> accept if distance <= Threshold
        TP = Maps * diag_maps * (distance <= Threshold) #Eucidean distance case --> accept if distance <= Threshold
        T  = Maps * diag_maps
        TPR = TP.sum() / T.sum()
        GAR[Ind] = TPR
          
        FR = Maps * diag_maps * (distance > Threshold) #Eucidean distance case --> reject if distance > Threshold 
        M  = Maps * diag_maps
        FRRi = FR.sum() / M.sum()
        FRR[Ind] = FRRi
          
        FP = (1 - Maps) * (distance <= Threshold) #Eucidean distance case --> accept if distance <= Threshold 
        N  = (1 - Maps);
        FPR = FP.sum() / N.sum()
        FAR[Ind] = FPR;

        Ind = Ind + 1 
    return GAR,FAR,FRR    

def computeDETdata(Maps, distance, ThresholdRange, diag_maps):
    GAR = np.zeros(len(ThresholdRange))
    FRR = np.zeros(len(ThresholdRange))
    FAR = np.zeros(len(ThresholdRange))

    Ind = 0;
    for Threshold in ThresholdRange:    
        
        TP = Maps * diag_maps * (distance > Threshold) #Cosine Similarity case --> accept if distance >= Threshold
        T  = Maps * diag_maps
        TPR = TP.sum() / T.sum()
        GAR[Ind] = TPR
          
        FR = Maps * diag_maps * (distance <= Threshold) #Cosine Similarity case --> reject if distance < Threshold 
        M  = Maps * diag_maps
        FRRi = FR.sum() / M.sum()
        FRR[Ind] = FRRi
          
        FP = (1 - Maps) * (distance >= Threshold) #Cosine Similarity case --> accept if distance >= Threshold 
        N  = (1 - Maps);
        FPR = FP.sum() / N.sum()
        FAR[Ind] = FPR;

        Ind += 1         
    return GAR,FAR,FRR    

def computeEER(fmr,fnmr,steps):
    # More advanced method by linear interpolation, using y=ax + b
    
    I = np.nanargmin(np.absolute((fnmr - fmr)));
    # eer = frr[np.nanargmin(np.absolute((frr - fpr)))]
    
    # If fmr and fnmr are equal->easy peasy
    if fmr[I] == fnmr[I]:
        #EER = fmr(I)*100;  As percentage
        EER = fmr[I]
        idx_thresh = I
        return EER
    
    #print(I)
    # Check if we are right of the 'zero-crossing'
    sign = lambda x: math.copysign(1, x)
    if (sign(fmr[I] - fnmr[I]) == -1) & (I>1):
        I = I - 1
    """
    #check the cross point of FPR and FRR! is that point at the border of index?
    #if yes, then decrease the stepsize.
    if (len(fmr) == I+1)|(len(fnmr) == I+1):
        I = I - 1
    """
    #print(I)
    #print(len(fmr))
    dy_fmr  = fmr[I+1] - fmr[I]     # Delta y for fmr
    dy_fnmr = fnmr[I+1] - fnmr[I]   # Delta y for fnmr
    dx      = steps[I+1] - steps[I] # Delta x
    
    a_fmr = dy_fmr/dx   # Estimated slope for fmr curve
    a_fnmr = dy_fnmr/dx # Estimated slope for fnmr curve
    b_fmr = fmr[I] - a_fmr*steps[I]    # Estimated offset for fmr curve
    b_fnmr = fnmr[I] - a_fnmr*steps[I] # Estimated offset for fnmr curve
    
    #idx_thresh = (b_fnmr - b_fmr)/(a_fmr - a_fnmr); % Determine intersection point
    #EER = a_fmr*idx_thresh + b_fmr;
    #EER = EER*100; % As percentage
    if (a_fmr - a_fnmr == 0):
        EER = (fmr[I] + fnmr[I])/2
        idx_thresh = I
    else:
        idx_thresh = (b_fnmr - b_fmr)/(a_fmr - a_fnmr) # Determine intersection point
        EER = a_fmr*idx_thresh + b_fmr
        
    return EER

def rgb_gaussian_filter(img, sigma):
    img2 = np.zeros(img.shape, dtype="uint8")
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    r1 = gaussian_filter(r, sigma=sigma)
    g1 = gaussian_filter(g, sigma=sigma)
    b1 = gaussian_filter(b, sigma=sigma)
    img2[:,:,0] = r1
    img2[:,:,1] = g1
    img2[:,:,2] = b1
    return img2            

def rgb_to_bw(img, thr):
    bw_img = img
    idx0 = (img < thr)
    idx1 = (img >= thr)
    bw_img[idx0] = 0
    bw_img[idx1] = 1
    return bw_img

def local_threshold_segmentation(img, img_gray, mask):
    blur = cv2.blur(img_gray, (3, 3)) # blur the image
    eyebrow_thr = filters.threshold_local(blur,block_size=75,offset=2)

    segmented = img_gray.copy()
    segmented[blur<=eyebrow_thr] = 255
    segmented[blur>eyebrow_thr] = 0
    
    backtorgb = cv2.cvtColor(segmented,cv2.COLOR_GRAY2RGB)
    
    segmented_mask = img.copy()
    segmented_mask[:,:,:]=0
    segmented_mask[mask==255] = backtorgb[mask==255]    
    
    return segmented_mask

def create_eyebrow_mask(image, left_eyebrow_shape, right_eyebrow_shape, convex_height, colors=None, alpha=0.5):
    
    overlay = image.copy()
    output = image.copy()
    overlay[:,:,:] = 0
    
    if colors is None:
        colors = [(255, 255, 255),(19, 199, 109), (79, 76, 240), (230, 159, 23),
            (168, 100, 168), (158, 163, 32),
            (163, 38, 32), (180, 42, 220)]
    
    pts_l = left_eyebrow_shape
    pts_r = right_eyebrow_shape
    
    w=pts_l.shape[0]
    pts2_l = np.zeros((w*2,2), dtype="int")
    pts2_r = np.zeros((w*2,2), dtype="int")
    
    for idx in np.arange(0,w,1):
        temp = pts_l[idx]   
        pts2_l[idx] =(temp[0],temp[1])
        pts2_l[2*w-1-idx] = (temp[0],temp[1]+convex_height) 

        temp = pts_r[idx]   
        pts2_r[idx] =(temp[0],temp[1])
        pts2_r[2*w-1-idx] = (temp[0],temp[1]+convex_height) 
        
    polygon = cv2.polylines(overlay, [pts2_l], True, colors[0], convex_height)
    polygon = cv2.polylines(overlay, [pts2_r], True, colors[0], convex_height)
    
    (x, y, w, h) = cv2.boundingRect(pts2_l)
    roi_left = image[y:y + h, x:x + w]
    
    (x, y, w, h) = cv2.boundingRect(pts2_r)
    roi_right = image[y:y + h, x:x + w]
    
    cv2.addWeighted(polygon, alpha, output, 1 - alpha, 0, output)
       
    return output, overlay, roi_right, roi_left


def eyebrows_detection(shape):
    # to np.array
    #shape = face_utils.shape_to_np(landmarks)
    left_eyebrow_shape = shape[17:22]
    right_eyebrow_shape = shape[22:27]
    
    return left_eyebrow_shape, right_eyebrow_shape    

# =============================================================================
# ============================= ArcFace =======================================
def ArcFace_crop(img, bboxes):
    img_size = img.shape[:2]
    bboxes = bboxes.reshape(-1, 2, 2) * img.shape[:2]
    bboxes = bboxes[...,::-1].astype(np.int32)
    img2 = img.copy()
    for bbox in bboxes:
        cv2.rectangle(img2, tuple(bbox[0]), tuple(bbox[1]), (0,255,0), 2)
        pts1 = tuple(bbox[0])
        pts2 = tuple(bbox[1])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(pts1[1], 0)
        bb[1] = np.maximum(pts1[0], 0)
        bb[2] = np.minimum(pts2[1], img_size[1])
        bb[3] = np.minimum(pts2[0], img_size[0])

        cropped = img[bb[0]:bb[2],bb[1]:bb[3],:]       
    return img2, cropped

def normalize_landmark_coor(img, bbox, landmark):
    # convert landmark coordinates relative to the whole image
    bbox = np.clip(bbox, 0, 1) # some values may be outside of range: [0,1]
    shape_img = img.shape[:2]
    bbox = np.reshape(bbox, [-1, 2]) * shape_img # fraction to pixel
    height, width = bbox[1] - bbox[0]
    landmark = np.reshape(landmark, [-1, 2])
    landmark = landmark * (height, width)
    landmark = landmark + bbox[0]
    landmark = landmark / img.shape[:2]
    return landmark

def crop_to_bbox(img, bbox):
    shape_img = img.shape[:2]
    bbox = np.clip(bbox, 0., 1.)
    bbox = np.reshape(bbox, [-1, 2]) * shape_img
    bbox = bbox.flatten().astype(np.int32)
    img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] # crop
    return img_cropped

def align(img, lm):
    lm = np.reshape(lm, [-1, 2])
    lm = lm * img.shape[:2]
    dist = np.linalg.norm(lm[0] - lm[-1]) * 1.5
    dist = dist.astype(np.int32)
    center = np.mean(lm, 0).astype(np.int32)
    bbox = [center - dist, center + dist]
    bbox = np.clip(bbox, 0, img.shape[:2])
    img = img[
        bbox[0][0]: bbox[1][0],
        bbox[0][1]: bbox[1][1],
        :]
    lm = lm - bbox[0]
    lm = lm[...,::-1]
    img_shape = img.shape[:2]
    # coordinates are taken from:
    # https://github.com/deepinsight/insightface/blob/master/src/common/face_preprocess.py
    # normalized to fraction from pixel values
    dst = np.array([
        [0.34191608, 0.46157411],
        [0.65653392, 0.45983393],
        [0.500225  , 0.64050538],
        [0.3709759 , 0.82469198],
        [0.631517  , 0.82325091]])
    dst =  dst * img_shape[::-1]
    tform = skitrans.SimilarityTransform()
    tform.estimate(lm, dst)
    M = tform.params[0:2,:]
    aligned = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]))
    return aligned
# ============================= End ArcFace ===================================
# =============================================================================