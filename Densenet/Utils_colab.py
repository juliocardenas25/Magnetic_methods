import h5py
import numpy as np
import cv2 as cv

def load_dataset_train_dev_test(hdf5_path):
	
	hdf5_file = h5py.File(hdf5_path, "r")
	
	X_train = np.array(hdf5_file["X_train"][:]) # your train set features
	X_dev = np.array(hdf5_file["X_dev"][:]) # your train set labels
	X_test = np.array(hdf5_file["X_test"][:]) # your train set labels

	Y_train = np.array(hdf5_file["Y_train"][:]) # your train set features
	Y_dev = np.array(hdf5_file["Y_dev"][:]) # your train set features
	Y_test = np.array(hdf5_file["Y_test"][:]) # your train set features

	
	return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

def gray_to_rgb(array):

	array_rgb = np.zeros(array.shape)
	array_rgb = np.repeat(array_rgb, 3, -1)

	for n_i in range(array.shape[0]):
		array_rgb[n_i,:,:,:] = cv.cvtColor(array[n_i,:,:,:], cv.COLOR_GRAY2RGB)

	return array_rgb.astype("float32")

def standardize_x_array(image):

	# initialize to array of zeros, with same shape as the image
	standardized_image = np.zeros(image.shape)

	for n_m in range(image.shape[0]):

		# subtract the mean from image_slice
		centered = image[n_m,:,:,:] - image[n_m,:,:,:].mean()

		if (image[n_m,:,:,:].std()) == 0 :

			centered_scaled = centered/(image[n_m,:,:,:].std() + 0.000001)

		else: 

			centered_scaled = centered/image[n_m,:,:,:].std()

	    # update  the slice of standardized image
	    # with the scaled centered and scaled image
		standardized_image[n_m,:,:,:] = centered_scaled

	return standardized_image.astype("float32")

def normalize_x_array(image, max, min):
  
  # initialize to array of zeros, with same shape as the image
  normalized_image = np.zeros(image.shape)

  for n_m in range(image.shape[0]):

    numerator = image[n_m,:,:,:] - min
    denominator = max - min
    
    if denominator == 0 :
      normalizing = numerator/(denominator + 0.000001)
    else: 
      normalizing = numerator/denominator

    normalized_image[n_m,:,:,:] = normalizing

  return normalized_image.astype("float32")


def parameters_to_use(Y_train, Y_dev, Y_test, flag):

	if flag == "rotation":

		Y_train = Y_train[:,0:1]
		Y_dev = Y_dev[:,0:1]
		Y_test = Y_test[:,0:1]

	elif flag == "parameters":

		Y_train = Y_train[:,2:4]
		Y_dev = Y_dev[:,2:4]
		Y_test = Y_test[:,2:4]


	return Y_train, Y_dev, Y_test


def Normalizing_Y_array(Y_array, max_r, max_d):

	Y_array_normalized = np.zeros(Y_array.shape)

	for n_m in range(Y_array.shape[0]):

		Y_array_normalized[n_m,0] = Y_array[n_m,0]/max_r
		Y_array_normalized[n_m,1] = Y_array[n_m,1]/max_d

	return Y_array_normalized.astype("float32")