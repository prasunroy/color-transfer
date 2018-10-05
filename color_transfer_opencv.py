# -*- coding: utf-8 -*-
"""
Color Transfer between Images.
An implementation of the paper "Color Transfer Between Images" by
Erik Reinhard, Michael Adhikhmin, Bruce Gooch and Peter Shirley (2001)
(http://www.cs.northwestern.edu/~bgooch/PDFs/ColorTransfer.pdf).

Created on Mon Oct  1 22:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/color-transfer

"""


# imports
import cv2
import numpy


# transfer color
def transfer_color(source_file, target_file, rescale=True):
    # read images as BGR
    source_bgr = cv2.imread(source_file, cv2.IMREAD_COLOR)
    target_bgr = cv2.imread(target_file, cv2.IMREAD_COLOR)
    
    # convert BGR to LAB
    source_lab = numpy.float32(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB))
    target_lab = numpy.float32(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB))
    
    # calculate mean and standard deviation of LAB images
    source_mu, source_sigma = _imstats(source_lab)
    target_mu, target_sigma = _imstats(target_lab)
    
    # ensure standard deviations to be non-zero to avoid divide-by-zero error
    source_sigma = numpy.where(source_sigma == 0,
                               numpy.ones_like(source_sigma, numpy.float32)*1e-4,
                               source_sigma)
    target_sigma = numpy.where(target_sigma == 0,
                               numpy.ones_like(target_sigma, numpy.float32)*1e-4,
                               target_sigma)
    
    # subtract mean of target from target (ref: eq. 10 in the paper)
    target_lab -= target_mu
    
    # scale target using standard deviations (ref: eq. 11 in the paper)
    target_lab *= (target_sigma / source_sigma)
    
    # add mean of source to target
    target_lab += source_mu
    
    # convert LAB to BGR
    result_bgr = cv2.cvtColor(numpy.uint8(target_lab), cv2.COLOR_LAB2BGR)
    if rescale:
        result_bgr = cv2.normalize(result_bgr, None, 0, 255,
                                   cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    return result_bgr


# calculate mean and standard deviation of an image along each channel
def _imstats(image):
    # reshape image from (M x N x 3) to (3 x MN) for vectorized operations
    image = numpy.float32(image).reshape(-1, 3).T
    
    # calculate mean
    mu = numpy.mean(image, axis=1, keepdims=False)
    
    # calculate standard deviation
    sigma = numpy.std(image, axis=1, keepdims=False)
    
    return (mu, sigma)
