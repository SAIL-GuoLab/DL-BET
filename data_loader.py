import os
import random
from glob import glob
from random import shuffle
import numpy as np
import nibabel as nib
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from sklearn.feature_extraction import image as sklearn_image
from misc import printProgressBar
import random
from scipy import ndarray
import skimage as sk

import sys
sys.path.append('./augmentation_class/')
from augmentation import Augmentation

#import pdb
torch.manual_seed(202005)
torch.cuda.manual_seed_all(202005)
np.random.seed(202005)
random.seed(202005)

class NiftiDataset(data.Dataset):
	def __init__(self, input_folder, GT_folder, mode, augment, augmentation_option):
		"""Initializes nifti file paths and preprocessing module."""

		# Define the directories where the input brain MRI and GT brain mask scans are stored.
		self.input_folder = input_folder
		self.GT_folder = GT_folder

		# Grab all the files in these directories.
		self.input_paths = list(np.sort(glob(input_folder + '*.nii.gz')))
		self.GT_paths = list(np.sort(glob(GT_folder + '*.nii.gz')))

		self.mode = mode
		self.augment = augment
		self.augmentation_option = augmentation_option

		# Report the number of files in each of the pre-gado, post-gado, and breast mask directories.
		print('\nInput (brain MRI) {} nifti file count: {}'.format(self.mode, len(self.input_paths)))
		print('GT (brain mask) {} nifti file count: {}'.format(self.mode, len(self.GT_paths)))

	def center_crop_or_pad(self, input_scan, desired_dimension):
		input_dimension = input_scan.shape
		#print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

		x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
		y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
		z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
		x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
		y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
		z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

		x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
		y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
		z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
		x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
		y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
		z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]


		output_scan = np.zeros(desired_dimension).astype(np.float32)

		output_scan[x_lowerbound_target : x_upperbound_target, \
					y_lowerbound_target : y_upperbound_target, \
					z_lowerbound_target : z_upperbound_target] = \
					input_scan[x_lowerbound_input: x_upperbound_input, \
								y_lowerbound_input: y_upperbound_input, \
								z_lowerbound_input: z_upperbound_input]

		return output_scan

	def augment_data(self, input_scan, GT_scan):
		if self.augmentation_option['noise_gaussian'] == 1:
			input_scan = Augmentation.noise_gaussian(input_scan)
		if self.augmentation_option['noise_Rayleigh'] == 1:
			input_scan = Augmentation.noise_Rayleigh(input_scan)
		if self.augmentation_option['contrast'] == 1:
			input_scan = Augmentation.contrast(input_scan)
		if self.augmentation_option['sharpness'] == 1:
			input_scan = Augmentation.sharpness(input_scan)
		if self.augmentation_option['equalization'] == 1:
			input_scan = Augmentation.equalization(input_scan)
		if self.augmentation_option['random_slope'] == 1:
			input_scan = Augmentation.random_slope(input_scan)
		if self.augmentation_option['random_offset'] == 1:
			input_scan = Augmentation.random_offset(input_scan)
		return input_scan, GT_scan

	def __getitem__(self, index):
		input_path = self.input_paths[index]
		input_nifti_scan = nib.load(input_path).get_fdata().astype(np.float32)
		# Scan-max normalization
		input_nifti_scan = input_nifti_scan / input_nifti_scan.max()
		
		'''
		CHECK HERE~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		Check for the correct string split and extension.
		'''
		filename = input_path.split('/')[-1]
		GT_path = self.GT_folder + filename.replace('pre_WH_N4corrected_iso60','new_gt_iso60-edited')
		GT_nifti_scan = nib.load(GT_path).get_fdata().astype(np.float32)
		
		if self.augment:
			input_nifti_scan, GT_nifti_scan = self.augment_data(input_nifti_scan, GT_nifti_scan)
			
		# Center crop as per network input requirements
		input_nifti_scan  = self.center_crop_or_pad(input_nifti_scan, (224, 288, 224))
		GT_nifti_scan  = self.center_crop_or_pad(GT_nifti_scan, (224, 288, 224))
		#input_nifti_scan  = self.center_crop_or_pad(input_nifti_scan, (224, 320, 192))
		#GT_nifti_scan  = self.center_crop_or_pad(GT_nifti_scan, (224, 320, 192))

		# Transpose to make depth the first dimension
		#input_nifti_scan = input_nifti_scan.transpose((-1,0,1))
		#GT_nifti_scan = GT_nifti_scan.transpose((-1,0,1))

		return input_nifti_scan, GT_nifti_scan

	def __len__(self):
		"""Returns the total number of nifti images."""
		return len(self.input_paths)

def get_loader_3D(input_folder, GT_folder, batch_size, num_workers = 1, mode = 'train', shuffle = True, augment = False, augmentation_option = None):
	"""Builds and returns Dataloader."""
	dataset = NiftiDataset(input_folder = input_folder, GT_folder = GT_folder, mode = mode, augment = augment, augmentation_option = augmentation_option)
	data_loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = False)
	return data_loader
