#!/usr/bin/env python3

import csv
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from more_itertools import locate

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.54907281448319933, -0.72074094308345071, 0.42314481842352314 ],
			"lookat" : [ -7.4165150225483982, -4.3692552972898397, 4.2418377265036487 ],
			"up" : [ -0.27778678941340029, 0.3201300269334113, 0.90573244696378663 ],
			"zoom" : 0.26119999999999988
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}



class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.25, ransac_n=3, num_iterations=50):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")
    point_cloud_original = o3d.io.read_point_cloud('../Ex2/factory_without_ground.ply')


    # ------------------------------------------
    # Execution
    # ------------------------------------------

    point_cloud = deepcopy(point_cloud_original) 
    print('before downsampling point cloud has ' + str(len(point_cloud.points)) + ' points')

    # Downsampling using voxel grid filter
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.1) 
    print('After downsampling point cloud has ' + str(len(point_cloud_downsampled.points)) + ' points')


    # Clustering

    cluster_idxs = list(point_cloud_downsampled.cluster_dbscan(eps=0.45, min_points=50, print_progress=True))

    print(cluster_idxs)
    print(type(cluster_idxs))

    possible_values = list(set(cluster_idxs))
    possible_values.remove(-1)
    print(possible_values)

    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points



    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))

    cloud_building = point_cloud_downsampled.select_by_index(largest_idxs)
    cloud_others = point_cloud_downsampled.select_by_index(largest_idxs, invert=True)

    cloud_others.paint_uniform_color([0,0,0.5])

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw
    entities = [cloud_building, cloud_others]

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('./factory_isolated.ply', cloud_building, write_ascii=False, compressed=False, print_progress=False)



if __name__ == "__main__":
    main()
