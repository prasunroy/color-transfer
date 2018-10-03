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
import numpy
from PIL import Image


# transfer color
def transfer_color(source_file, target_file, rescale=True):
    # read images as RGB
    source_rgb = Image.open(source_file).convert('RGB')
    target_rgb = Image.open(target_file).convert('RGB')
    
    # convert RGB to LAB
    source_lab = _rgb2lab(numpy.uint8(source_rgb))
    target_lab = _rgb2lab(numpy.uint8(target_rgb))
    
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
    
    # convert LAB to RGB
    result_rgb = _lab2rgb(target_lab)
    if rescale:
        result_rgb = numpy.uint8(_rescale(result_rgb))
    else:
        result_rgb = numpy.uint8(result_rgb)
    result_rgb = Image.fromarray(result_rgb)
    
    return result_rgb


# convert RGB to LAB color space
def _rgb2lab(array):
    # verify dimension of the input array
    assert len(array.shape) == 3 and array.shape[2] == 3, \
    'Input array needs to be a RGB image.'
    
    # initialize transformation matrices (ref: eq. 4 and eq. 6 in the paper)
    T1_RGB2LMS = numpy.float32([[0.3811, 0.5783, 0.0402],
                                [0.1967, 0.7244, 0.0782],
                                [0.0241, 0.1288, 0.8444]])
    T2_LMS2LAB = numpy.float32([[1., 1., 1.],
                                [1., 1., -2.],
                                [1., -1., 0.]])
    T3_LMS2LAB = numpy.float32([[1./numpy.sqrt(3.), 0., 0.],
                                [0., 1./numpy.sqrt(6.), 0.],
                                [0., 0., 1./numpy.sqrt(2.)]])
    
    # reshape array from (M x N x 3) to (3 x MN) for vectorized operations
    RGB = numpy.float32(array).reshape(-1, 3).T
    
    # convert RGB to LMS (ref: eq. 4 in the paper)
    LMS = numpy.matmul(T1_RGB2LMS, RGB)
    
    # convert LMS to logarithmic space (ref: eq. 5 in the paper)
    LMS = numpy.where(LMS == 0, numpy.ones_like(LMS, numpy.float32)*1e-4, LMS)
    LMS = numpy.log10(LMS)
    
    # convert LMS to LAB (ref: eq. 6 in the paper)
    LAB = numpy.matmul(T3_LMS2LAB, numpy.matmul(T2_LMS2LAB, LMS))
    
    # reshape array from (3 x MN) to (M x N x 3)
    LAB = LAB.T.reshape(array.shape[0], array.shape[1], 3)
    
    return LAB


# convert LAB to RGB color space
def _lab2rgb(array):
    # verify dimension of the input array
    assert len(array.shape) == 3 and array.shape[2] == 3, \
    'Input array needs to be a LAB image.'
    
    # initialize transformation matrices (ref: eq. 8 and eq. 9 in the paper)
    T1_LAB2LMS = numpy.float32([[numpy.sqrt(3)/3., 0., 0.],
                                [0., numpy.sqrt(6)/6., 0.],
                                [0., 0., numpy.sqrt(2)/2.]])
    T2_LAB2LMS = numpy.float32([[1., 1., 1.],
                                [1., 1., -1.],
                                [1., -2., 0.]])
    T3_LMS2RGB = numpy.float32([[4.4679, -3.5873, 0.1193],
                                [-1.2186, 2.3809, -0.1624],
                                [0.0497, -0.2439, 1.2045]])
    
    # reshape array from (M x N x 3) to (3 x MN) for vectorized operations
    LAB = numpy.float32(array).reshape(-1, 3).T
    
    # convert LAB to LMS (ref: eq. 8 in the paper)
    LMS = numpy.matmul(T2_LAB2LMS, numpy.matmul(T1_LAB2LMS, LAB))
    LMS = 10. ** LMS
    
    # convert LMS to RGB (ref: eq. 9 in the paper)
    RGB = numpy.matmul(T3_LMS2RGB, LMS)
    
    # reshape array from (3 x MN) to (M x N x 3)
    RGB = RGB.T.reshape(array.shape[0], array.shape[1], 3)
    
    return RGB


# calculate mean and standard deviation of an image along each channel
def _imstats(image):
    # reshape image from (M x N x 3) to (3 x MN) for vectorized operations
    image = numpy.float32(image).reshape(-1, 3).T
    
    # calculate mean
    mu = numpy.mean(image, axis=1, keepdims=False)
    
    # calculate standard deviation
    sigma = numpy.std(image, axis=1, keepdims=False)
    
    return (mu, sigma)


# rescale image to [0., 255.] range
def _rescale(image):
    image = numpy.float32(image)
    image = 255. * (image - image.min()) / (image.max() - image.min())
    return image
