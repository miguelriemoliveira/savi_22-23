#!/usr/bin/env python3

import csv
from copy import deepcopy
from random import randint
from turtle import color

import cv2
import numpy as np
from scipy.optimize import least_squares
from models import ImageMosaic


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    # Two images, query (q) and target (t)
    q_path = '../images/machu_pichu/query_warped.png'
    q_image = cv2.imread(q_path)
    q_gui = deepcopy(q_image)
    q_gray = cv2.cvtColor(q_image,cv2.COLOR_BGR2GRAY)
    q_win_name = 'Query Image'

    t_path = '../images/machu_pichu/target.png'
    t_image = cv2.imread(t_path)
    t_gui = deepcopy(t_image)
    t_gray = cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)
    t_win_name = 'Target Image'

    image_mosaic = ImageMosaic(q_image, t_image) # created the class

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    x0 = [image_mosaic.q_scale, image_mosaic.q_bias, image_mosaic.t_scale, image_mosaic.t_bias]
    result = least_squares(image_mosaic.objectiveFunction, x0, verbose=2)

    image_mosaic.draw()
    cv2.waitKey(0)

    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()
