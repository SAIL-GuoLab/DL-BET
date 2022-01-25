import os
from glob import glob
import numpy as np
import nibabel as nib
import time
import datetime
import torch
import torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, KLDivLoss, BCELoss
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from network import init_weights, ResAttU_Net3D
from evaluation import compute_MAE, compute_MSE, compute_PSNR, compute_NCC
import csv
from tqdm import tqdm
from scipy import ndimage
import pandas as pd
from shutil import copyfile
from matplotlib import pyplot as plt
from robust_loss_pytorch import AdaptiveLossFunction
from evaluation import compute_DICE
torch.manual_seed(202001)
torch.cuda.manual_seed_all(202001)
np.random.seed(202001)
#import pdb

class Solver(object):
	def __init__(self, config, train_loader, validation_loader, test_loader):
		"""Initialize our deep learning model."""
		# Data loader
		self.train_loader = train_loader
		self.validation_loader = validation_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.initialization = config.initialization
		self.optimizer = None
		self.img_ch = config.img_ch
		self.UnetLayer = config.UnetLayer
		self.first_layer_numKernel = config.first_layer_numKernel
		self.corregistration = config.corregistration
		self.device = torch.device('cuda: %d' % config.cuda_idx)

		# Hyper-parameters
		self.initial_lr = config.initial_lr
		self.current_lr = config.initial_lr
		if 'loss_function_lr' in config.__class__.__dict__:
			self.loss_function_lr = config.loss_function_lr
		if 'adaptive_lr' in config.__class__.__dict__:
			self.adaptive_lr = config.adaptive_lr

		if 'clipped_gradient' in config.__class__.__dict__:
			if config.clipped_gradient != 'NA':
				self.clipped_gradient = config.clipped_gradient
				self.gradient_max_norm = config.gradient_max_norm

		self.optimizer_choice = config.optimizer_choice
		if config.optimizer_choice == 'Adam':
			self.beta1 = config.beta1
			self.beta2 = config.beta2
		elif config.optimizer_choice == 'SGD':
			self.momentum = config.momentum
		else:
			print('No such optimizer available')

		# Loss Function
		self.reference_loss_function_exists = False
		if config.loss_function_name == 'MSE':
			self.loss_function_name = 'MSE'
			self.loss_function = MSELoss()
		elif config.loss_function_name == 'SmoothL1':
			self.loss_function_name = 'SmoothL1'
			self.loss_function = SmoothL1Loss()
		elif config.loss_function_name == 'BCELoss':
			self.loss_function_name = 'BCELoss'
			self.loss_function = BCELoss()
		elif config.loss_function_name == 'CVPR_Adaptive_loss':
			self.loss_function_name = 'CVPR_Adaptive_loss'
			self.reference_loss_function = MSELoss()
			self.reference_loss_function_exists = True
			self.loss_function = AdaptiveLossFunction(num_dims = 1, float_dtype = np.float32, alpha_init = 2, alpha_hi = 3.5, device = self.device)

		# Also add the KL divergence
		self.histogram_bins = int(128)
		self.KL_divergence = KLDivLoss(reduction = 'batchmean')

		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Early stop or not
		self.early_stop = config.early_stop

		# Path
		self.model_saving_path = config.model_saving_path
		self.prediction_path = config.predictions_path
		self.loss_history_path = config.loss_histories_path
		self.test_result_comparison_path = config.test_result_comparison_path

		self.test_input_folder = config.test_input_folder
		self.test_GT_folder = config.test_GT_folder
		self.test_input_paths = list(np.sort(glob(self.test_input_folder + '*.nii.gz')))
		self.test_GT_paths = list(np.sort(glob(self.test_GT_folder + '*.nii.gz')))

		self.mode = config.mode
		self.augment = config.augment
		self.augmentation_option = config.augmentation_option
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build our deep learning model."""
		if self.model_type == 'ResAttU_Net3D':
			self.unet = ResAttU_Net3D(UnetLayer = self.UnetLayer, img_ch = 1, output_ch = 1, first_layer_numKernel = self.first_layer_numKernel)

		if self.optimizer_choice == 'Adam':
			if self.loss_function_name == 'CVPR_Adaptive_loss':
				self.optimizer = optim.Adam(list(self.unet.parameters()), self.initial_lr, [self.beta1, self.beta2])
				self.loss_function_optimizer = optim.Adam(list(self.loss_function.parameters()), self.loss_function_lr, [self.beta1, self.beta2])
			else:
				self.optimizer = optim.Adam(list(self.unet.parameters()), self.initial_lr, [self.beta1, self.beta2])
		elif self.optimizer_choice == 'SGD':
			if self.loss_function_name == 'CVPR_Adaptive_loss':
				self.optimizer = optim.SGD(list(self.unet.parameters()), self.initial_lr, self.momentum)
				self.loss_function_optimizer = optim.SGD(list(self.loss_function.parameters()), self.loss_function_lr, self.momentum)
			else:
				self.optimizer = optim.SGD(list(self.unet.parameters()), self.initial_lr, self.momentum)
		else:
			pass

		if self.adaptive_lr == True:
			print('Initializing adaptive learning rate... ')
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.85, patience = 2, verbose = 2)
		else:
			self.scheduler = None

		if self.initialization != 'NA':
			init_weights(self.unet, init_type = self.initialization)
		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))
		
	def to_data(self, x):
		"""Convert tensor to numpy."""
		if torch.cuda.is_available():
			x = x.cpu().detach().numpy()
		return x

	# Redefine the 'update_lr' function (R&R)
	def update_lr(self, optimizer, new_lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr          

	def backprop_and_optimize(self, loss, epoch):
		"""Backpropagate the loss and optimize the parameters."""
		# Step 1. Reset the gradient
		self.optimizer.zero_grad()
		if self.loss_function_name == 'CVPR_Adaptive_loss':
			self.loss_function_optimizer.zero_grad()
		# Step 2. Backpropagate
		loss.backward()

		#grads = []
		#for param in self.unet.parameters():
		#	grads.append(param.grad.view(-1))
		#grads = torch.cat(grads)
		#grads = np.sqrt(np.sum((grads.cpu().detach().numpy())**2))
		#print('Model Gradient', grads)

		if not self.clipped_gradient is None:
			self.gradient_clip(epoch)

		# Step 4. Update the parameters.
		self.optimizer.step()
		if self.loss_function_name == 'CVPR_Adaptive_loss' and epoch > 0:
			self.loss_function_optimizer.step()

	def gradient_clip(self, epoch):
		try:
			self.gradient_max_norm
			if self.gradient_max_norm is None:
				raise TypeError('No value has been set for the max gradient norm.')
		except:
			raise NameError('Undefined: max gradient norm.')

		clip_grad_norm_(self.unet.parameters(), self.gradient_max_norm)
		
		if self.loss_function_name == 'CVPR_Adaptive_loss' and epoch > 0:            
			clip_grad_norm_(self.loss_function.parameters(), self.gradient_max_norm)

	def train(self):
		"""Train our deep learning model."""
		#====================================== Training ===========================================#
		#===========================================================================================#

		best_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'best'))
		last_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'last'))
		print('The best model weight will be saved as {} \nThe last model weight will be saved as {}'.format(best_unet_path, last_unet_path))
		
		# U-Net Train
		# Train loss history (R&R)
		train_loss_history = []
		# Validation loss history (R&R)
		validation_loss_history = []

		# Create the path to save the loss history csv files.
		for this_directory in [self.loss_history_path, self.test_result_comparison_path]:
			if not os.path.exists(this_directory):
				os.makedirs(this_directory)

		# save current configs
		copyfile('./deep_learning_model/config_settings.ini', self.loss_history_path + 'config_settings.ini')
		copyfile('./deep_learning_model/config.py', self.loss_history_path + 'config.py')

		if os.path.isfile(best_unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(best_unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type, best_unet_path))
		else:
			# Train for Encoder
			print('Start training. The initial learning rate is: {}'.format(self.initial_lr))
            
			# Write the first line of the train and validation loss history csv file.
			with open(os.path.join(self.loss_history_path, 'train_and_validation_history.csv'), 'a', \
					encoding = 'utf-8', newline= '') as f:
				wr = csv.writer(f)
				wr.writerow(['Mode', 'Current Epoch', 'Total Epoch', 'Batch Size', 'Metric', 'Loss', 'Reference Metric', 'Loss'])
				f.close()

			for epoch in range(self.num_epochs):
				self.unet.train(True)
				train_epoch_loss_sum = 0; validation_epoch_loss_sum = 0; reference_epoch_loss_sum = 0; validation_reference_epoch_loss_sum = 0; #epoch_KL_divergence = 0;

				length = 0
				loop = tqdm(total=len(self.train_loader), position = 0, leave = False)

				for batch, (input_scan, GT_scan) in enumerate(self.train_loader):
					loop.set_description("Training...".format(batch))
					loop.update(1)
					# Forward all the four images to CUDA GPU.
					input_scan = input_scan.to(self.device)
					GT_scan = GT_scan.to(self.device)

					# Reshape the scans to 5-dimensional so that they can get fed to the conv3d layer. (R&R)
					# The new shape has to be (batch_size, num_channels, img_dim1, img_dim2, img_dim3).
					input_scan = input_scan[:, np.newaxis, :, :, :]
					GT_scan = GT_scan[:, np.newaxis, :, :, :]
					
					if input_scan.shape[0] != self.batch_size:
						print('\rNote: The shape for train batch {} is {}.'.format(batch, input_scan.shape))

					# Compute the Prediction.
					# Use the network to make predictions with input_scan as input.
					Prediction = self.unet(input_scan)
					
					'''
					if batch == 0:
						plt.subplot(1,3,1)
						plt.imshow(np.squeeze(input_scan.cpu().detach().numpy())[:, :, input_scan.shape[-1]//2], clim = [0,1], cmap = 'jet')
						plt.axis('off')
						plt.subplot(1,3,2)
						plt.imshow(np.squeeze(GT_scan.cpu().detach().numpy())[:, :, input_scan.shape[-1]//2], clim = [0,1], cmap = 'jet')
						plt.axis('off')
						plt.subplot(1,3,3)
						plt.imshow(np.squeeze(Prediction.cpu().detach().numpy())[:, :, input_scan.shape[-1]//2], clim = [0,1], cmap = 'jet')
						plt.axis('off')
						plt.show()
					'''

					# Flatten the Prediction and GT_scan as vectors. Do the same to the masks as appropriate.
					Prediction_vector = torch.flatten(Prediction)
					GT_scan_vector = torch.flatten(GT_scan)
					
					# Compute the loss for this batch.
					if self.loss_function_name == 'CVPR_Adaptive_loss':
						train_loss = torch.mean(self.loss_function.lossfun((Prediction_vector - GT_scan_vector)[:,None]))
						reference_loss = self.reference_loss_function(Prediction_vector, GT_scan_vector).item()
					elif self.loss_function_name == 'BCELoss':
						train_loss = self.loss_function(nn.Sigmoid()(Prediction_vector), GT_scan_vector)
					else:
						train_loss = self.loss_function(Prediction_vector, GT_scan_vector)
				
					# Add the loss of this batch to the loss of this epoch.
					train_epoch_loss_sum += train_loss.item(); length += 1
					if self.loss_function_name == 'CVPR_Adaptive_loss':
							reference_epoch_loss_sum += reference_loss

					# Backprop + optimize, with gradient clipping
					self.backprop_and_optimize(train_loss, epoch)

					# Print the train loss every quarter of an epoch.
					#if batch > 0 and batch % (len(self.train_loader) // 4)== 0:
						#print('\r\33[2K[Training] Epoch [{}/{}], Batch: {}, Batch size: {}, Average {} Error: {}'.format(epoch + 1, self.num_epochs, batch, \
																#self.batch_size, self.loss_function_name, train_epoch_loss_sum/length))

						#if self.loss_function_name == 'CVPR_Adaptive_loss':
							#print('\r\33[2KCVPR Adaptive loss: MSE loss={:03f}  loss={:03f}  alpha={:03f}  scale={:03f}'.format(reference_epoch_loss_sum/length, train_loss.data, self.loss_function.alpha()[0,0].data, self.loss_function.scale()[0,0].data)) 

					# Delete used variables to clean up the memory.
					del batch, input_scan, GT_scan, Prediction, Prediction_vector, GT_scan_vector, train_loss
					# Empty cache to free up memory at the end of each batch.
					#torch.cuda.empty_cache()

				# Normalize the train loss over the length of the epoch (number of images in this epoch).
				train_epoch_loss = train_epoch_loss_sum/length

				# Print the log info
				print('\r\33[2K[Training] Epoch [%d/%d], Train Loss: %.6f' % (epoch + 1, self.num_epochs, train_epoch_loss))

				# Append train loss to train loss history (R&R)
				train_loss_history.append(train_epoch_loss)
				with open(os.path.join(self.loss_history_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '') as f:
					wr = csv.writer(f)
					wr.writerow(['Training', '%d' % (epoch + 1), '%d' % (self.num_epochs), '%d' % (self.batch_size), \
                         	     '%s' % self.loss_function_name, '%.6f' % train_epoch_loss, \
                         	     '%s' % (self.reference_loss_function if self.reference_loss_function_exists else 'NA'), \
                         	     '%.6f' % (reference_epoch_loss_sum/length if self.reference_loss_function_exists else 0)])
					f.close()

				#===================================== Validation ====================================#
				"""Validate the deep learning network at the current training stage."""
				self.unet.train(False)
				self.unet.eval()

				length = 0

				for batch, (input_scan, GT_scan) in enumerate(self.validation_loader):
					input_scan = input_scan.to(self.device)
					GT_scan = GT_scan.to(self.device)

					input_scan = input_scan[:, np.newaxis, :, :, :]
					GT_scan = GT_scan[:, np.newaxis, :, :, :]
                    
					if input_scan.shape[0] != self.batch_size:
						print('\rNote: The shape for validation batch {} is {}.'.format(batch, input_scan.shape))

					Prediction = self.unet(input_scan)
					
					Prediction_vector = torch.flatten(Prediction)
					GT_scan_vector = torch.flatten(GT_scan)

					if self.loss_function_name == 'CVPR_Adaptive_loss':
						validation_loss = torch.mean(self.loss_function.lossfun((Prediction_vector - GT_scan_vector)[:,None]))
					elif self.loss_function_name == 'BCELoss':
						validation_loss = self.loss_function(nn.Sigmoid()(Prediction_vector), GT_scan_vector)
					else:
						validation_loss = self.loss_function(Prediction_vector, GT_scan_vector)

					validation_epoch_loss_sum += validation_loss.item(); length += 1
					if self.loss_function_name == 'CVPR_Adaptive_loss':
						validation_reference_epoch_loss_sum += reference_loss

					del batch, input_scan, GT_scan, Prediction, Prediction_vector, GT_scan_vector, validation_loss
					#torch.cuda.empty_cache() 
                    
				# Normalize the validation loss.
				validation_epoch_loss = validation_epoch_loss_sum/length

				# Define the decisive score of the network as 1 - validation_epoch_loss.
				unet_score = 1. - validation_epoch_loss
				print('Current learning rate: {}'.format(self.current_lr))

				print('[Validation] Epoch [%d/%d] Validation Loss: %.6f' % (epoch + 1, self.num_epochs, validation_epoch_loss))
				if self.loss_function_name == 'CVPR_Adaptive_loss':
					print('\rMSE loss={:03f}'.format(validation_reference_epoch_loss_sum/length))

				# Append validation loss to train loss history (R&R)
				validation_loss_history.append(validation_epoch_loss)
				
				with open(os.path.join(self.loss_history_path, 'train_and_validation_history.csv'), 'a', \
                         encoding = 'utf-8', newline= '') as f:
					wr = csv.writer(f)
					wr.writerow(['Validation', '%d' % (epoch + 1), '%d' % (self.num_epochs), '%d' % (self.batch_size), \
                             	 '%s' % self.loss_function_name, '%.6f' % validation_epoch_loss, \
                         	     '%s' % (self.reference_loss_function if self.reference_loss_function_exists else 'NA'), \
                         	     '%.6f' % (validation_reference_epoch_loss_sum/length if self.reference_loss_function_exists else 0)])
					f.close()

				# Create paths if not exist (R&R)
				for this_directory in [self.model_saving_path, self.loss_history_path, self.test_result_comparison_path]:
					if not os.path.exists(this_directory):
						os.makedirs(this_directory)

				# Save the model at every epoch.
				all_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'epoch', str(epoch + 1).zfill(2)))
				this_unet = self.unet.state_dict()
				torch.save(this_unet, all_unet_path)
      
				# Make sure we save the best and last unets.
				if epoch == 0:
					best_unet_score = unet_score - 0.1
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.6f' % (self.model_type, best_unet_score))
					torch.save(best_unet, best_unet_path)
				if (epoch == self.num_epochs - 1):
					last_unet = self.unet.state_dict()
					torch.save(last_unet, last_unet_path)

				#pdb.set_trace()
				# Adaptive Learning Rate (R&R)
				if self.scheduler != None:
					self.scheduler.step(validation_epoch_loss)

				# Early stop (R&R)
				if (self.early_stop == True):
					if (len(validation_loss_history) > 9):
						if (np.mean(validation_loss_history[-10:-5]) <= np.mean(validation_loss_history[-5:])):
							print('Validation loss stop decreasing. Stop training.')
							last_unet = self.unet.state_dict()
							torch.save(last_unet, last_unet_path)
							break      

		del self.unet
		try:
			del best_unet
		except:
			print('Cannot delete the variable "best_unet": variable does not exist.')
        
		return train_loss_history, validation_loss_history

	def test(self, which_unet = 'best', save_prediction_npy = False, save_prediction_nifti = True, save_GT_nifti = True):
		"""Test encoder, generator and discriminator."""

		#======================================= Test ====================================#
		#=================================================================================#
		best_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'best'))
		last_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'last'))

		if which_unet == 'best':
			self.unet.load_state_dict(torch.load(best_unet_path))
		elif which_unet == 'last':
			self.unet.load_state_dict(torch.load(last_unet_path))
		elif str(which_unet).isdigit() == True:
			all_unet_path = os.path.join(self.model_saving_path, '%s-%s-%.4f-%s-%d-%s%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'epoch', str(which_unet).zfill(2)))
			self.unet.load_state_dict(torch.load(all_unet_path))
		else:
			print('Input argument which_unet is invalid. Has to be "best" or "last" or an integer representing the epoch.')

		# Make the directory for the current prediction.
		if not os.path.exists(self.prediction_path):
			os.makedirs(self.prediction_path)

		# We only allow saving the gado uptake ground truth (GT) if 'save_prediction_nifti' is True.
		if save_prediction_nifti == False and save_GT_nifti == True:
			save_prediction_nifti = True
			print('The parameter "save_prediction_nifti" is set to "True". We only allow saving the GT when we are also saving the predictions.')

		self.unet.train(False)
		self.unet.eval()
		length = 0
		test_epoch_loss_sum = 0
        
		# Evaluation metrics.
		mean_absolute_error = 0
		mean_squared_error = 0
		peak_SNR = 0
		structural_similarity = 0
		normalized_cross_correlation = 0
		#ROI_correlation_coefficient = 0
        
		test_input_paths = list(np.sort(glob(self.test_input_folder + '*.nii.gz')))
		test_GT_paths = list(np.sort(glob(self.test_GT_folder + '*.nii.gz')))

		with torch.no_grad():
			for batch, (input_scan, GT_scan) in enumerate(self.test_loader):
				# Check the batch size of the test loader. We only support "1".
				test_batch_size = input_scan.shape[0]
				if test_batch_size == int(1) and batch == 0:
					print('The prediction will be made 1 slice at a time. The image shape is: {}.'.format(input_scan.shape))
				elif test_batch_size != int(1):
					raise ValueError('Check the batch_size in test_loader. We currently only support "1".')
				del test_batch_size


				'''
				# CHECK HERE FOR SPLITTING ~~~~~~~~~~~~~~~~~~~~ !!!!!!!!!!!!!!!!!!!!!!!
				'''
				# Find the corresponding image filename to name the generated nifti file.
				corresponding_input_scan_path = test_input_paths[batch]
				file_name_with_extension = corresponding_input_scan_path.split('/')[-1]
				filename = file_name_with_extension[:-7]

				# Create a folder for the current subject if we would like to save the slices in npy format.
				if (save_prediction_npy == True):
					if not os.path.exists(self.prediction_path + 'subject_%s/' % filename):
						os.mkdir(self.prediction_path + 'subject_%s/' % filename)

				if (save_prediction_nifti == True):
					corresponding_input_nifti = nib.load(corresponding_input_scan_path)
					current_scan_affine = corresponding_input_nifti.affine
					current_scan_header = corresponding_input_nifti.header
					del corresponding_input_nifti

				input_scan = input_scan.to(self.device)
				GT_scan = GT_scan.to(self.device)

				input_scan = input_scan[:, np.newaxis, :, :, :]
				GT_scan = GT_scan[:, np.newaxis, :, :, :]

				Prediction = self.unet(input_scan)
				Prediction_vector = torch.flatten(Prediction)
				GT_scan_vector = torch.flatten(GT_scan)

				if self.loss_function_name == 'CVPR_Adaptive_loss':
					test_loss = torch.mean(self.loss_function.lossfun((Prediction_vector - GT_scan_vector)[:,None]))
				elif self.loss_function_name == 'BCELoss':
					test_loss = self.loss_function(nn.Sigmoid()(Prediction_vector), GT_scan_vector)
				else:
					test_loss = self.loss_function(Prediction_vector, GT_scan_vector)

				test_epoch_loss_sum += test_loss.item(); length += 1

				# Use the region within the brain mask to evaluate the performance.
				#brain_mask_vector = torch.flatten(GT_scan)
				#mean_absolute_error += compute_MAE(Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy(), GT_scan_vector[brain_mask_vector == 1].cpu().detach().numpy())
				#mean_squared_error += compute_MSE(Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy(), GT_scan_vector[brain_mask_vector == 1].cpu().detach().numpy())
				#peak_SNR += compute_PSNR(Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy(), GT_scan_vector[brain_mask_vector == 1].cpu().detach().numpy())
				#normalized_cross_correlation += compute_NCC(Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy(), GT_scan_vector[brain_mask_vector == 1].cpu().detach().numpy())


				# Actually, let's save out the scan after the sigmoid. (2020/07/24 midnight)

				# Save the prediction scan as numpy arrays if we would like to have numpy output.
				if save_prediction_npy == True:
					np_image = np.squeeze(nn.Sigmoid()(Prediction).cpu().detach().numpy())
					np.save(self.prediction_path + 'patient_%s/' % filename + filename + '_predictedBM.npy', np_image)

				# Save the prediction scan if we would like to have nifti output.
				if save_prediction_nifti == True:
					current_prediction_scan = np.squeeze(nn.Sigmoid()(Prediction).cpu().detach().numpy())
					current_prediction_scan_nifti = nib.Nifti1Image(current_prediction_scan, current_scan_affine, current_scan_header)
					nib.save(current_prediction_scan_nifti, self.prediction_path + filename + '_predictedBM.nii.gz')
					dice_coef = compute_DICE(current_prediction_scan, GT_scan.cpu().detach().numpy())
					print(f'{filename} dice coefficient: ', dice_coef)

					del current_prediction_scan, current_prediction_scan_nifti, current_scan_affine, current_scan_header
					del input_scan, GT_scan, Prediction, Prediction_vector, GT_scan_vector, test_loss
					# Empty cache to free up memory at the end of each batch.
					#torch.cuda.empty_cache()

		test_epoch_loss = test_epoch_loss_sum / length

		#mean_absolute_error = mean_absolute_error / length
		#mean_squared_error = mean_squared_error / length
		#peak_SNR = peak_SNR / length
		#structural_similarity = structural_similarity / length
		#normalized_cross_correlation = normalized_cross_correlation / length

		#print('Model type: %s, Optimizer: %s, Initial learning rate: %.4f, Loss function: %s, Batch size: %d, Best or last: %s, Test Loss: %.6f' \
		#		% (self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, which_unet, test_epoch_loss))
		#print('MAE: %.6f, MSE: %.6f, PSNR: %.6f, SSIM: %.6f, MSSSIM: %.6f, NCC: %.6f' \
		#			% (mean_absolute_error, mean_squared_error, peak_SNR, structural_similarity, normalized_cross_correlation))

		# Store the test result together with all other ones for comparison
		# If the file doesn't exist or is empty, write the header.
		#if os.path.exists(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv')):
		#	df = pd.read_csv(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'))
		#	if df.empty:
		#		with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
		#			wr = csv.writer(f)
		#			wr.writerow(['Corregistration or not', 'Model type', 'Initialization', 'Standardization', 'Optimizer', 'Initial learning rate', 'Loss function', 'Batch size', 'U-Net layers', 'Kernels in first layer', 'Best or last', 'Test loss', \
		#					 	 'Mean Absolute Error', 'Mean Squared Error', 'Peak Signal-to-Noise Ratio', 'Structural Similarity', 'Normalized Cross Correlation'])
		#			f.close()
		#else:
		#	with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
		#		wr = csv.writer(f)
		#		wr.writerow(['Corregistration or not', 'Model type', 'Initialization', 'Standardization', 'Optimizer', 'Initial learning rate', 'Loss function', 'Batch size', 'U-Net layers', 'Kernels in first layer', 'Best or last', 'Test loss', \
		#				 	 'Mean Absolute Error', 'Mean Squared Error', 'Peak Signal-to-Noise Ratio', 'Structural Similarity', 'Normalized Cross Correlation'])
		#		f.close()

        # Always fill in the test result for this trial.
		#with open(os.path.join(self.test_result_comparison_path, 'test_result_comparison.csv'), 'a', encoding = 'utf-8', newline= '') as f:
		#	wr = csv.writer(f)
		#	wr.writerow([self.corregistration, self.model_type, self.initialization, self.standardization, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, self.UnetLayer, self.first_layer_numKernel, which_unet, '%.6f' % test_epoch_loss, \
		#				 '%.6f' % mean_absolute_error, '%.6f' % mean_squared_error, '%.6f' % peak_SNR, \
		#				 '%.6f' % structural_similarity, '%.6f' % normalized_cross_correlation])
		#	f.close()
