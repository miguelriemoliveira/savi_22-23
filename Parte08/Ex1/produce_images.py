#!/usr/bin/env python3

import csv
from copy import deepcopy
from random import randint
from turtle import color

import cv2
import numpy as np


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    # Two images, query (q) and target (t)
    q_path = '../images/machu_pichu/2.png'
    q_image = cv2.imread(q_path)
    q_gui = deepcopy(q_image)
    q_gray = cv2.cvtColor(q_image,cv2.COLOR_BGR2GRAY)
    q_win_name = 'Query Image'

    t_path = '../images/machu_pichu/1.png'
    t_image = cv2.imread(t_path)
    t_gui = deepcopy(t_image)
    t_gray = cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
    t_win_name = 'Target Image'

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    # Create sift detection object
    sift = cv2.SIFT_create(nfeatures=200)

    # Detect keypoints in both images
    q_key_points, q_des = sift.detectAndCompute(q_gray,None) # SIFT features
    t_key_points, t_des = sift.detectAndCompute(t_gray,None) # SIFT features

    # Match features (FLANN based matching)
    index_params = dict(algorithm = 1, trees = 15)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    best_two_matches = flann.knnMatch(q_des, t_des,k=2)

    # Create a list containing only the best matches, and use David Lowe's ratio test to compute the uniqueness of a match
    matches = []
    for best_two_match in best_two_matches:
        best_match = best_two_match[0]
        second_best_match = best_two_match[1]

        # David Lowe's test
        if  best_match.distance < 0.3 * second_best_match.distance:
            matches.append(best_match)

    # Compute homography
    # First we need to create the np.arrays of size (n_pts, 1, 2) to feed into the findhomography

    num_pts = len(matches)
    q_pts_array = np.ndarray((num_pts, 1, 2), dtype=np.float32) # create a np array of the query points to feed find homography
    t_pts_array = np.ndarray((num_pts, 1, 2), dtype=np.float32) # create a np array of the query points to feed find homography

    for idx_match, match in enumerate(matches):

        q_idx = match.queryIdx
        q_x = q_key_points[q_idx].pt[0]
        q_y = q_key_points[q_idx].pt[1]
        q_pts_array[idx_match, 0, 0] = q_x
        q_pts_array[idx_match, 0, 1] = q_y
        
        t_idx = match.trainIdx
        t_x = t_key_points[t_idx].pt[0]
        t_y = t_key_points[t_idx].pt[1]
        t_pts_array[idx_match, 0, 0] = t_x
        t_pts_array[idx_match, 0, 1] = t_y

    M, mask = cv2.findHomography(q_pts_array, t_pts_array ,cv2.RANSAC)



    # Warp q image to move it to the t image coordinate frame
    q_h, q_w, _ = q_image.shape
    t_h, t_w, _ = t_image.shape

    # When q_image is inside t_image
    stitched_image_h = t_h
    stitched_image_w = t_w

    q_image_warped = cv2.warpPerspective(q_image, M,(stitched_image_w, stitched_image_h))
    q_image_warped = q_image_warped[:,:,0:3] # remove fourth channel

    # stitch images

    # Alt 1: Average all the pixels
    # stitched_image = ((t_image.astype(float) + q_image_warped.astype(float))/2).astype(np.uint8) # initialization

    # Alt 2: Use q pixels in  overlapping region
    # Create an overlapping mask
    # overlap_mask = q_image_warped > 0
    # stitched_image = deepcopy(t_image)
    # stitched_image[overlap_mask] = q_image_warped[overlap_mask]

    # Alt 3: Average only overlapping pixels
    overlap_mask = q_image_warped > 0
    stitched_image = deepcopy(t_image)
    stitched_image[overlap_mask] = ((q_image_warped[overlap_mask].astype(float) + stitched_image[overlap_mask].astype(float) ) / 2).astype(np.uint8)




    # Visualize ----------------------

    # Draw keypoints 
    for idx, key_point in enumerate(q_key_points):
        x = int((key_point.pt[0]))
        y = int((key_point.pt[1]))
        color = (randint(0,255), randint(0,255), randint(0,255))
        cv2.circle(q_gui, (x,y), 30, color, 3)

    for idx, key_point in enumerate(t_key_points):
        x = int((key_point.pt[0]))
        y = int((key_point.pt[1]))
        color = (randint(0,255), randint(0,255), randint(0,255))
        cv2.circle(t_gui, (x,y), 30, color, 3)

    # Show the matches image
    matches_image = cv2.drawMatches(q_image, q_key_points,
                                    t_image, t_key_points, matches,None)

    cv2.namedWindow(q_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(q_win_name, 600, 400)
    cv2.imshow(q_win_name, q_gui)

    cv2.namedWindow(t_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(t_win_name, 600, 400)
    cv2.imshow(t_win_name, t_gui)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', matches_image)

    cv2.namedWindow('q_image_warped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('q_image_warped', 600, 400)
    cv2.imshow('q_image_warped', q_image_warped)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 600, 400)
    cv2.imshow('mask', (overlap_mask*255).astype(np.uint8))

    cv2.namedWindow('stitched_image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('stitched_image', 600, 400)
    cv2.imshow('stitched_image', stitched_image)


    cv2.waitKey(0)

    cv2.imwrite('../images/machu_pichu/query_warped.png', q_image_warped)
    cv2.imwrite('../images/machu_pichu/target.png', t_image)



    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()
