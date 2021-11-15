#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from matplotlib.patches import Rectangle

def R_to_V(R):

	V = np.round(((4)*(np.pi)*(R**3)) / 3, decimals=3)

	return V

def V_to_R(V):

	R = np.cbrt((3*V) / (4*(np.pi)))

	return R

def real_to_pixel(array):

	conversion = array * (zmax/(2*map_lenght))

	return conversion

def pixel_to_real(array):

	conversion = array * ((2*map_lenght)/zmax)

	return conversion


def grid(zmax, map_lenght):

	X = np.linspace(-map_lenght, map_lenght, zmax).reshape(zmax,1)
	Y = np.linspace(-map_lenght, map_lenght, zmax).reshape(1,zmax)

	return X,Y


def latitudes_array(lat_min, lat_max, lat_frequency):

	lat_array = np.arange(lat_min, lat_max, lat_frequency)
	lat_array_rad = lat_array * (np.pi/180)

	I = np.arctan(2*np.tan(lat_array_rad)) # Inclinaison

	return lat_array, lat_array_rad, I


def r_V_h_array(r_min, r_max, r_frequency, h_min, h_max, h_frequency, zmax, Br, ksi):

	r_array = np.arange(r_min, r_max, r_frequency)
	V_array = R_to_V(r_array)
	h_array = np.arange(h_min, h_max, h_frequency)

	n_examples = r_array.shape[0] * h_array.shape[0]

	Parameters_array = np.zeros((n_examples, 3))

	i_p = 0

	for i_r in r_array:
		for i_h in h_array:
			Parameters_array[i_p,0] = i_r
			Parameters_array[i_p,1] = i_h
			Parameters_array[i_p,2] = R_to_V(i_r)*ksi*Br # Dipole moment [A.m^2]

			i_p += 1

	m = Br * V_array * ksi

	return r_array, V_array, h_array, n_examples, Parameters_array, m


def X_array_raw(Br, m, I, X, Y, h, H_capteur_bas, H_capteur_haut, N_lat, examples, zmax, n_parameters, lat_array_rad, Parameters_array):

	X_array_raw = np.zeros((N_lat, examples, zmax, zmax)) # Array with magnetic induction values

	Parameters_array_raw = np.zeros((N_lat, examples, n_parameters)) # Array with volume and depth values

	i = 0

	for i_lat_rad in lat_array_rad:

		Anomalie = None

		r = np.sqrt(np.add(np.add(X**2, Y**2), (h**2).reshape(h.shape[0], 1, 1)))

		costeta = np.divide(((X * np.cos(I[i])) - (np.multiply(h.reshape(h.shape[0], 1, 1), np.sin(I[i])))), r)

		Anomalie = np.outer(m.reshape(m.shape[0],1,1), np.divide(3*(costeta**2) - 1, r**3)).reshape((examples,zmax,zmax))

		X_array_raw[i, :, :, :] = Anomalie

		Parameters_array_raw[i,:,:] = Parameters_array[:,:]

		i += 1

	return X_array_raw, Parameters_array_raw

def Plot_X_array_raw(map_lenght, height, width, example_to_plot, lat_array, X_data_array_raw, Parameters_array_raw, n_model_aug, n_dipoles, rows, cols, zmax):

	n_lat = lat_array.shape[0]

	X = np.linspace(-map_lenght, map_lenght, zmax)
	Y = np.linspace(-map_lenght, map_lenght, zmax)

	print(f"Plotting model: {example_to_plot}")

	fig, axs = plt.subplots(rows, cols, figsize=(height, width))
	fig.subplots_adjust(hspace = 0, wspace=0.1)

	axs = axs.ravel()

	for lat_i in range(n_lat):

		cp = axs[lat_i].contourf(X, Y, X_data_array_raw[lat_i,example_to_plot,:,:], levels = 16, cmap='rainbow')
		axs[lat_i].set_xlabel('Position X (m)')
		axs[lat_i].set_ylabel('Position Y (m)')
		axs[lat_i].set_xticks(np.arange(-25, 25, step=2))
		axs[lat_i].set_yticks(np.arange(-25, 25, step=2))

		title = lat_array[lat_i]
		axs[lat_i].set_title(f'Latitude = {title}°')

		clb = fig.colorbar(cp, ax=axs[lat_i])
		clb.set_label('Magnetic induction (nT)', labelpad=15, y=0.5, rotation=270)

	plt.tight_layout()

def plot_contour_raw(map_lenght, height_2, width_2, example_to_plot, X_data_array_raw, outer_x, outer_y, box_anomaly, lat_array):

	n_examples = X_data_array_raw.shape[0]
	zmax = X_data_array_raw.shape[2]
	n_lat = X_data_array_raw.shape[0]

	pixel_to_real = ((map_lenght*2)/zmax)
	real_to_pixel = zmax/(map_lenght*2)

	if example_to_plot == 'random':

		example_to_plot = int(np.random.randint(0, n_examples, 1))

	model = X_data_array_raw[example_to_plot,:,:]

	X = np.linspace(-map_lenght, map_lenght, zmax)
	Y = np.linspace(-map_lenght, map_lenght, zmax)

	model_x = np.roll(model, outer_x, axis=1)
	model_y = np.roll(model_x, outer_y, axis=0)

	print(f"Plotting model: {example_to_plot}")

	if outer_x != 0:
		print(f"outer_X: {int(outer_x + (zmax/2))}, outer_Y: {int(outer_y + (zmax/2))}")

	rows, cols = n_lat, 2

	fig, axs = plt.subplots(rows, cols, figsize=(height_2, width_2))
	#fig.subplots_adjust(hspace = 0, wspace=0)

	axs = axs.ravel()

	for lat_i in range(n_lat):

		x0 = box_anomaly[lat_i,example_to_plot,0] + (outer_x * pixel_to_real)
		y0 = box_anomaly[lat_i,example_to_plot,1] + (outer_y * pixel_to_real)
		bwidth = box_anomaly[lat_i,example_to_plot,2]
		bheight = box_anomaly[lat_i,example_to_plot,3]

		rect_real = Rectangle((x0,y0),bwidth,bheight, edgecolor='r', facecolor="none")

		x0_2 = (x0 + map_lenght) * real_to_pixel
		y0_2 = (y0 + map_lenght) * real_to_pixel
		bwidth_2 = (bwidth) * real_to_pixel
		bheight_2 = (bheight) * real_to_pixel

		rect_pixel = Rectangle((x0_2,y0_2),bwidth_2,bheight_2, edgecolor='r', facecolor="none")

		cp_1 = axs[(2*lat_i)].contourf(X, Y, X_data_array_raw[lat_i,example_to_plot,:,:], levels = 40, cmap='seismic')
		axs[(2*lat_i)].set_xlabel('Position X (m)')
		axs[(2*lat_i)].set_ylabel('Position Y (m)')
		axs[(2*lat_i)].set_xticks(np.arange(-25, 25, step=2))
		axs[(2*lat_i)].set_yticks(np.arange(-25, 25, step=2))
		axs[(2*lat_i)].add_patch(rect_real)

		title = lat_array[lat_i]
		axs[(2*lat_i)].set_title(f'Latitude = {title}°')

		#clb = fig.colorbar(cp_1)
		#clb.set_label('Magnetic induction (nT)', labelpad=15, y=0.5, rotation=270)

		cp_2 = axs[(2*lat_i)+1].imshow(X_data_array_raw[lat_i,example_to_plot,:,:])
		axs[(2*lat_i)+1].set_xlabel('Position X (m)')
		axs[(2*lat_i)+1].set_ylabel('Position Y (m)')
		axs[(2*lat_i)+1].set_xticks(np.arange(0, 416, step=20))
		axs[(2*lat_i)+1].set_yticks(np.arange(0, 416, step=20))
		axs[(2*lat_i)+1].add_patch(rect_pixel)

		title = lat_array[lat_i]
		axs[(2*lat_i)+1].set_title(f'Latitude = {title}°')

	plt.tight_layout()



def Plot_X_data(map_lenght, height_2, width_2, example_to_plot, X_data_array_raw, Position_f, box_anomaly, n_dipoles, inclination, n_lat_to_plot):

	n_examples = X_data_array_raw.shape[1]
	zmax = X_data_array_raw.shape[2]
	bbox = box_anomaly
	position = Position_f
	n_lat = X_data_array_raw.shape[2]

	pixel_to_real = ((map_lenght*2)/zmax)
	real_to_pixel = zmax/(map_lenght*2)

	model = X_data_array_raw[:,example_to_plot,:,:]

	X = np.linspace(-map_lenght, map_lenght, zmax)
	Y = np.linspace(-map_lenght, map_lenght, zmax)

	print(f"Plotting model: {example_to_plot}")

	rows, cols = n_lat_to_plot, 2

	fig, axs = plt.subplots(rows, cols, figsize=(height_2, width_2))
	#fig.subplots_adjust(hspace = 0, wspace=0)

	axs = axs.ravel()

	for lat_i in range(n_lat_to_plot):

		rect_pixel = []
		rect_real = []

		for n_ii in range(n_dipoles):

			x0 = (bbox[lat_i,example_to_plot,(5*n_ii)+1] - (zmax/2)) * pixel_to_real
			y0 = (bbox[lat_i,example_to_plot,(5*n_ii)+2] - (zmax/2)) * pixel_to_real
			bwidth = bbox[lat_i,example_to_plot,(5*n_ii)+3] * pixel_to_real
			bheight = bbox[lat_i,example_to_plot,(5*n_ii)+4] * pixel_to_real

			rect_real.append(Rectangle((x0,y0),bwidth,bheight, edgecolor='r', facecolor="none"))

			x0_2 = bbox[lat_i,example_to_plot,(5*n_ii)+1]
			y0_2 = bbox[lat_i,example_to_plot,(5*n_ii)+2]
			bwidth_2 = bbox[lat_i,example_to_plot,(5*n_ii)+3]
			bheight_2 = bbox[lat_i,example_to_plot,(5*n_ii)+4]

			print(f"Dipole_{n_ii}: x0: {np.round(x0_2,1)}, y0: {np.round(y0_2,1)}, w: {round(bwidth_2,1)}, h:{round(bheight_2,1)}")

			rect_pixel.append(Rectangle((x0_2,y0_2),bwidth_2,bheight_2, edgecolor='r', facecolor="none"))

		cp_1 = axs[(2*lat_i)].contourf(X, Y, X_data_array_raw[lat_i,example_to_plot,:,:], levels = 18, cmap='seismic')
		axs[(2*lat_i)].set_xlabel('Position X (m)')
		axs[(2*lat_i)].set_ylabel('Position Y (m)')
		axs[(2*lat_i)].set_xticks(np.arange(-25, 25, step=2))
		axs[(2*lat_i)].set_yticks(np.arange(-25, 25, step=2))
		for n_iii in range(n_dipoles):
			axs[(2*lat_i)].add_patch(rect_real[n_iii])

		title = inclination[lat_i]
		axs[(2*lat_i)].set_title(f'Latitude = {title}°')

		#clb = fig.colorbar(cp_1)
		#clb.set_label('Magnetic induction (nT)', labelpad=15, y=0.5, rotation=270)

		cp_2 = axs[(2*lat_i)+1].imshow(X_data_array_raw[lat_i,example_to_plot,:,:], origin= "lower")
		axs[(2*lat_i)+1].set_xlabel('Position X (m)')
		axs[(2*lat_i)+1].set_ylabel('Position Y (m)')
		axs[(2*lat_i)+1].set_xticks(np.arange(0, 128, step=20))
		axs[(2*lat_i)+1].set_yticks(np.arange(0, 128, step=20))
		for n_iii in range(n_dipoles):
			axs[(2*lat_i)+1].add_patch(rect_pixel[n_iii])

		title = inclination[lat_i]
		axs[(2*lat_i)+1].set_title(f'Latitude = {title}°')

	plt.tight_layout()


def Plot_X_data_final(map_lenght, height_2, width_2, example_to_plot, X_data_array_raw, Position_f, box_anomaly, n_dipoles, lat_array_f, n_lat):

	n_examples = X_data_array_raw.shape[0]
	zmax = X_data_array_raw.shape[1]
	bbox = box_anomaly
	position = Position_f

	pixel_to_real = ((map_lenght*2)/zmax)
	real_to_pixel = zmax/(map_lenght*2)

	X = np.linspace(-map_lenght, map_lenght, zmax)
	Y = np.linspace(-map_lenght, map_lenght, zmax)

	print(f"Plotting model: {example_to_plot}")

	fig, axs = plt.subplots(1, 2, figsize=(height_2, width_2))
	#fig.subplots_adjust(hspace = 0, wspace=0)

	rect_pixel = []
	rect_real = []

	for n_ii in range(n_dipoles):

		x0 = (bbox[example_to_plot,n_ii,1] - (zmax/2)) * pixel_to_real
		y0 = (bbox[example_to_plot,n_ii,2] - (zmax/2)) * pixel_to_real
		bwidth = bbox[example_to_plot,n_ii,3] * pixel_to_real
		bheight = bbox[example_to_plot,n_ii,4] * pixel_to_real

		rect_real.append(Rectangle((x0,y0),bwidth,bheight, edgecolor='r', facecolor="none"))

		x0_2 = bbox[example_to_plot,n_ii,1]
		y0_2 = bbox[example_to_plot,n_ii,2]
		bwidth_2 = bbox[example_to_plot,n_ii,3]
		bheight_2 = bbox[example_to_plot,n_ii,4]

		rect_pixel.append(Rectangle((x0_2,y0_2),bwidth_2,bheight_2, edgecolor='r', facecolor="none"))

		print(f"Dipole_{n_ii}: x0: {position[example_to_plot,n_ii,0]}, y0: {position[example_to_plot,n_ii,1]}, w: {int(bwidth_2)}, h:{int(bheight_2)}")



	cp_1 = axs[(0)].contourf(X, Y, X_data_array_raw[example_to_plot,:,:], levels = 18, cmap='seismic')
	axs[(0)].set_xlabel('Position X (m)')
	axs[(0)].set_ylabel('Position Y (m)')
	axs[(0)].set_xticks(np.arange(-25, 25, step=2))
	axs[(0)].set_yticks(np.arange(-25, 25, step=2))
	for n_iii in range(n_dipoles):
		axs[(0)].add_patch(rect_real[n_iii])

	title = lat_array_f[example_to_plot, 0]
	axs[(0)].set_title(f'Latitude = {title}°')

	#clb = fig.colorbar(cp_1)
	#clb.set_label('Magnetic induction (nT)', labelpad=15, y=0.5, rotation=270)


	cp_2 = axs[1].imshow(X_data_array_raw[example_to_plot,:,:], origin= "lower")
	axs[1].set_xlabel('Position X (m)')
	axs[1].set_ylabel('Position Y (m)')
	axs[1].set_xticks(np.arange(0, 128, step=20))
	axs[1].set_yticks(np.arange(0, 128, step=20))
	for n_iii in range(n_dipoles):
		axs[1].add_patch(rect_pixel[n_iii])

	title = lat_array_f[example_to_plot, 0]
	axs[1].set_title(f'Latitude = {title}°')

	plt.tight_layout()


def padding_array(pad, padder, X_data_array_raw):

	##### Padding model to avoid border effect ######

	def pad_with(vector, pad_width, iaxis, kwargs):
		pad_value = kwargs.get('padder', padder)
		vector[:pad_width[0]] = pad_value
		vector[-pad_width[1]:] = pad_value

	###### Padding my models #########

	pad = int(X_data_array_raw.shape[2]/2)
	padder = 0

	X_data_pad = np.zeros((X_data_array_raw.shape[0], X_data_array_raw.shape[1], X_data_array_raw.shape[2]+(2*pad), X_data_array_raw.shape[2]+(2*pad)))

	for i in range(0, X_data_array_raw.shape[0], 1):

		for n_pad in range(0, X_data_array_raw.shape[1], 1):

			X_data_pad[i,n_pad,:,:] = np.pad(X_data_array_raw[i, n_pad,:,:], pad, pad_with)

	return X_data_pad

def X_array(N_models, N_dipoles, N_lat, zmax, examples, X_data_pad, Parameters_array_raw, conversion):

	X_data_array = np.zeros((N_models, N_dipoles, N_lat, examples , zmax, zmax))
	Parameters_array = np.zeros((N_models, N_dipoles, N_lat, examples , Parameters_array_raw.shape[2]))
	Position_array = np.zeros((N_models, N_dipoles, N_lat, examples , N_dipoles*Parameters_array_raw.shape[2]))
	Lat_counting = np.zeros((N_models, N_dipoles, N_lat, examples , 1))
	N_dipoles_counting = np.zeros((N_models, N_dipoles, N_lat, examples , 1))
	ID = np.zeros((N_models*N_dipoles*N_lat*examples , 1))

	limit = (zmax/2) - 1

	i_ID = 0

	for n_models_i in range(0, N_models, 1):

		print(str(f"X model {n_models_i} ready"))

		for n_dipoles_i in range(0, N_dipoles, 1):

			X_data_y = np.zeros((N_lat,examples, zmax*2, zmax*2))

			Position_array_to_add = np.zeros((N_lat,examples, Parameters_array_raw.shape[2]))

			for i_lat in range(0, N_lat, 1):

				for n_examples in range(0, examples, 1):

					X_data_x = None

					position_x = np.random.randint(-limit, limit)
					position_y = np.random.randint(-limit, limit)

					Position_array_to_add[i_lat, n_examples, 0] = position_x
					Position_array_to_add[i_lat, n_examples, 1] = position_y

					X_data_x = np.roll(X_data_pad[i_lat,n_examples,:,:], position_x)
					X_data_y[i_lat,n_examples,:,:] = np.roll(X_data_x, position_y, axis=0)

					Lat_counting[n_models_i,n_dipoles_i,i_lat,n_examples, 0] = i_lat
					N_dipoles_counting[n_models_i,n_dipoles_i,i_lat,n_examples, 0] = n_dipoles_i

					ID[i_ID,0] = i_ID

					i_ID +=1

			if n_dipoles_i == 0 :

				X_data_array[n_models_i,n_dipoles_i,:,:,:,:] = X_data_y[:, :, 64:192,  64:192]

				Position_array[n_models_i,n_dipoles_i:,:,:,(n_dipoles_i)*2:(n_dipoles_i+1)*2] = (Position_array_to_add[:, :, :]) * conversion

				Parameters_array[n_models_i,n_dipoles_i,:,:,:] = Parameters_array_raw[:,:,:]

			else:

				X_data_array[n_models_i,n_dipoles_i,:,:,:,:] = X_data_y[:,:,  64:192,  64:192] + X_data_array[n_models_i,n_dipoles_i-1,:,:,:,:]

				Position_array[n_models_i,n_dipoles_i:,:,:,(n_dipoles_i)*2:(n_dipoles_i+1)*2] = (Position_array_to_add[:, :, :]) * conversion

				Parameters_array[n_models_i, n_dipoles_i, :,:,:] = Parameters_array_raw[:,:,:]

	return X_data_array, Parameters_array, Position_array, Lat_counting, N_dipoles_counting, ID


def Y_array(N_models, N_dipoles, N_lat, zmax, n_examples, Parameters_array, Position_array):

	Y_data_array = np.zeros((N_models, N_dipoles, N_lat, n_examples, zmax, zmax))

	for n_models_i in range(0, N_models, 1):

		print(str(f"Y_model {n_models_i} ready"))

		for i_dipoles in range(0, N_dipoles, 1):

			for i_lat in range(0, N_lat, 1):

				for i_examples in range(0, n_examples, 1):

					Y_r = V_to_R(Parameters_array[n_models_i,i_dipoles,i_lat,i_examples,0])

					for i_circle in range(0, i_dipoles+1, 1):

						Y_pos_x = (Position_array[n_models_i,i_dipoles, i_lat, i_examples, (i_circle*2)]) / (50/zmax)
						Y_pos_y = (Position_array[n_models_i,i_dipoles, i_lat, i_examples, (i_circle*2)+1]) / (50/zmax)

						rr, cc = draw.circle((zmax/2)+Y_pos_y, (zmax/2)+Y_pos_x, Y_r)

						Y_data_array[n_models_i,i_dipoles,i_lat,i_examples,rr,cc,0] = i_lat

	return Y_data_array


def gaussian_noise_models(mean, var, sigma, n_models_with_noise, zmax, X_data_array, Y_data_array):

	gaussian = np.random.normal(mean, sigma, (zmax,zmax))

	X_data_array_with_noise = np.zeros((X_data_array.shape))

	for i_noise_models in range(0, n_models_with_noise, 1):

		X_data_array_with_noise[i_noise_models,:,:,:,:,:] = X_data_array[i_noise_models,:, :, :, :, :] + gaussian

	Y_data_array_with_noise = Y_data_array[:n_models_with_noise,:,:,:,:,:,:]

	return X_data_array_with_noise, Y_data_array_with_noise

def noise_models_visualization(zmax, map_lenght, height, width, example_to_plot, lat_array, X_data_array_with_noise_final, Lat_counting_noise_final, N_dipoles_counting_noise_final):

	X = np.linspace(-map_lenght, map_lenght, zmax)
	Y = np.linspace(-map_lenght, map_lenght, zmax)

		# Adjust the size of your images
	plt.figure(figsize=(height, width))

	if example_to_plot == 'random':

		example_to_plot = int(np.random.randint(0, (X_data_noise.shape[0]-6), 1))

	else:

		example_to_plot = example_to_plot

	i=0

	if parameter_to_plot == 'Volume':

		parameter = 0

	elif parameter_to_plot == 'Depth':

		parameter = 1

	# Iterate and plot random images
	for i_model_n in range(example_to_plot, example_to_plot+6, 1):

		#if i_c < 1:

		plt.subplot(3, 3, i + 1)
		plt.contourf(X, Y, X_data_noise[i_model_n,:,:], levels = 1, cmap='rainbow')
		plt.axis('on')
		plt.xlabel('Position X (m)')
		plt.ylabel('Position Y (m)')

		plt.title('Lat: ' + str(int(Lat_counting_noise_final[i_model_n])) + 'N_dipoles: ' + N_dipoles_counting_noise_final[i_model_n])

		clb = plt.colorbar()
		clb.set_label(parameter_to_plot + '(m^3)', labelpad=15, y=0.5, rotation=270)

		i+=1

	# Adjust subplot parameters to give specified padding
	plt.tight_layout()

