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


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")
    point_cloud = o3d.io.read_point_cloud('../data/Factory/factory.ply')

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    print('Starting plane detection')
    plane_model, inlier_idxs = point_cloud.segment_plane(distance_threshold=0.3, 
                                                    ransac_n=3,
                                                    num_iterations=100)
    [a, b, c, d] = plane_model
    print('Plane equation: ' + str(a) +  ' x + ' + str(b) + ' y + ' + str(c) + ' z + ' + str(d) + ' = 0' )

    inlier_cloud = point_cloud.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) # paints the plane in red
    outlier_cloud = point_cloud.select_by_index(inlier_idxs, invert=True)

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('./factory_without_ground.ply', outlier_cloud, write_ascii=False, compressed=False, print_progress=False)

if __name__ == "__main__":
    main()
