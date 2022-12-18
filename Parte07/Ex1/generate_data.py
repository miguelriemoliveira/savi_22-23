#!/usr/bin/env python3

import csv
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    plt.figure()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")

    print('Created a figure.')
    # plt.waitforbuttonpress()
    # plt.show()


    # ------------------------------------------
    # Execution
    # ------------------------------------------
    pts = {'xs': [], 'ys': []}

    while True:

        plt.plot(pts['xs'], pts['ys'], 'rx', linewidth=2, markersize=12)
        pt = plt.ginput(1)

        if not pt:
            print('Terminated')
            break
            
        print('pt = ' + str(pt))

        pts['xs'].append(pt[0][0])
        pts['ys'].append(pt[0][1])

        print('pts = ' + str(pts))

    # ------------------------------------------
    # Termination
    # ------------------------------------------

    file = open('pts.pkl', 'wb')
    pickle.dump(pts, file)        
    file.close()


if __name__ == "__main__":
    main()
