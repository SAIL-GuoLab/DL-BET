import numpy as np

# Mean Absolute Error (average L1 distance).
def compute_MAE(Prediction_vector, Target_vector):
	MAE = abs(Prediction_vector - Target_vector).sum() / Target_vector.size
	return MAE

# Mean Squared Error (average L2 distance).
def compute_MSE(Prediction_vector, Target_vector):
	MSE = ((Prediction_vector - Target_vector)**2).sum() / Target_vector.size
	return MSE

# Peak Signal-to-Noice Ratio.
def compute_PSNR(Prediction_vector, Target_vector):
	MSE = ((Prediction_vector - Target_vector)**2).sum() / Target_vector.size
	MaxI = np.max(Target_vector) - np.min(Target_vector)
	PSNR = 10 * np.log10(MaxI**2 / MSE)
	return PSNR

# Normalized Cross Correlation.
def compute_NCC(Prediction_vector, Target_vector):
	NCC = (np.multiply(Prediction_vector, Target_vector) / (np.std(Prediction_vector) * np.std(Target_vector))).sum() / Target_vector.size
	return NCC

def compute_DICE(Prediction_vector, Target_vector):
	smooth = 1
	product = np.multiply(Prediction_vector, Target_vector)
	intersection = np.sum(product)
	coeffient = (2 * intersection + smooth) / (np.sum(Prediction_vector) + np.sum(Target_vector) + smooth)
	return coeffient