# coding=utf-8
# Augmentation
# Define the augmentation methods as multiple functions within the same class.

import math
import random
import numpy as np
from scipy.signal import convolve
from skimage.exposure import equalize_hist

class Augmentation:
	def random_slope(scan, slope_min = 0.8, slope_max = 1.2):
		slope = random.uniform(slope_min, slope_max)
		return np.float64(scan) * slope

	def random_offset(scan, offset_min = -0.1, offset_max = 0.1):
		offset = random.uniform(offset_min, offset_max)
		return np.float64(scan) + offset

	def noise_gaussian(scan, mean = 0.0, std = 0.03):
		noise = np.random.normal(mean, std, scan.shape)
		return np.float64(scan) + noise - np.mean(noise)

	def noise_Rayleigh(scan, scale = 0.045):
		noise = np.random.rayleigh(scale, scan.shape)
		return np.float64(scan) + noise - np.mean(noise)

	def contrast(scan, scale = None, factor = None):
		'''Implementation of contrast stretching.'''
		original_scan = np.float64(scan)
		if scale is None:
			scale = np.random.uniform(0.8, 1.2)
		if factor is None:
			factor = np.random.uniform(0.0, 1.0)

		degenerated_scan = np.float64(scan) * scale
		degenerated_scan[degenerated_scan > original_scan.max()] = original_scan.max()

		return blend(degenerated_scan, original_scan, factor)

	def sharpness(scan, factor = None):
		original_scan = np.float64(scan)

		kernel = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], \
						   [[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]], \
						   [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])/21.0

		degenerated_scan = convolve(np.float64(scan), kernel, 'same')

		if factor is None:
			factor = np.random.uniform(0.0, 1.0)

		return blend(degenerated_scan, original_scan, factor)

	def equalization(scan, factor = None):
		original_scan = np.float64(scan)

		if factor is None:
			factor = np.random.uniform(0.0, 1.0)

		degenerated_scan = equalize_hist(np.float64(scan))
		return blend(degenerated_scan, original_scan, factor)

def blend(scan1, scan2, factor):
	if factor < 0.0 or factor > 1.0:
		raise ValueError('Error: blend factor must be between 0 and 1.')

	return np.float64(scan1) * (1.0 - factor) + np.float64(scan2) * factor