# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation


# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.utils.argtools import memoized
from tensorpack import (TowerTrainer,
                        ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import freeze_variables

# Tensorflow 
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc, SeparateGANTrainer, MultiGPUGANTrainer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe
from sklearn.metrics.cluster import adjusted_rand_score

###############################################################################
SHAPE = 256
BATCH = 1
TEST_BATCH = 100
EPOCH_SIZE = 100
NB_FILTERS = 16  # channel size

DIMX  = 1024
DIMY  = 1024
DIMZ  = 3
DIMC  = 1

MAX_LABEL = 320
###############################################################################
def magnitute_central_difference(image, name=None):
	from tensorflow.python.framework import ops
	from tensorflow.python.ops import math_ops
	with ops.name_scope(name, 'magnitute_central_difference'):
		ndims = image.get_shape().ndims
		Gx = tf.zeros_like(image)
		Gy = tf.zeros_like(image)

		if ndims == 3:
			pass
		elif ndims == 4:
			# The input is a batch of image with shape:
			# [batch, height, width, channels].

			# Calculate the difference of neighboring pixel-values.
			# The image are shifted one pixel along the height and width by slicing.
			padded_img1 = tf.pad(image, paddings=[[0,0], [1,1], [0,0], [0,0]], mode="REFLECT")
			padded_img2 = tf.pad(image, paddings=[[0,0], [0,0], [1,1], [0,0]], mode="REFLECT")
			# padded_img3 = tf.pad(image, paddings=[[1,1], [0,0], [0,0], [0,0]], mode="REFLECT")
			
			Gx = 0.5*(padded_img1[:,:-2,:,:] - padded_img1[:,2:,:,:])
			Gy = 0.5*(padded_img2[:,:,:-2,:] - padded_img2[:,:,2:,:])
			# Gz = 0.5*(padded_img3[:-2,:,:,:] - padded_img3[2:,:,:,:])
			# grad = tf.sqrt(tf.add_n([tf.pow(Gx,2),tf.pow(Gy,2),tf.pow(Gz,2)]))
			# return grad
		else:
			raise ValueError('\'image\' must be either 3 or 4-dimensional.')

		grad = tf.sqrt(tf.add(tf.square(Gx),tf.square(Gy))) # okay
		return grad

		loss_img = cvt2tanh(loss_img)
		return loss_val, loss_img
###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)

###############################################################################
# Utility function for scaling 
def cvt2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def cvt2imag(x, maxVal = 255.0, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	return (x / 2.0 + 0.5) * maxVal

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv0', chan, padding='SAME')
				.Conv2D('conv1', chan/2, padding='SAME')
				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
				.InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=1, stride=1):
	with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.75)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Deconv2D('deconv_i', chan, stride=1) 
			.residual('res2_', chan, first=True)
			.Deconv2D('deconv_o', chan, stride=2) 
			# .Dropout('drop', 0.75)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=1):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		e3 = Dropout('dr', e3, 0.5)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd

@auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=LeakyReLU)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e0 = Dropout('dr', e0, 0.5)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret


###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
		self.dtype      = dtype
		self.imageDir   = imageDir
		self.labelDir   = labelDir
		self._size      = size
		self.isTrain    = isTrain
		self.isValid    = isValid

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.imageDir + '/*.tif')
		labels = glob.glob(self.labelDir + '/*.tif')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		labels = natsorted(labels)


		#
		# Pick the image over size 
		#
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			rand_index = np.random.randint(0, len(images))
			image_p = skimage.io.imread(images[rand_index])
			# membr_p = skimage.io.imread(labels[rand_index])
			label_p = skimage.io.imread(labels[rand_index])
			membr_p = label_p.copy()


			#
			# Pick randomly a tuple of training instance
			#
			rand_image = np.random.randint(0, len(images))
			rand_membr = np.random.randint(0, len(images))
			rand_label = np.random.randint(0, len(images))


			# image_u = skimage.io.imread(images[rand_image])
			# membr_u = skimage.io.imread(labels[rand_membr])
			# label_u = skimage.io.imread(labels[rand_label])
			image_u = image_p.copy() #skimage.io.imread(images[rand_index])
			membr_u = membr_p.copy() #skimage.io.imread(labels[rand_index])
			label_u = label_p.copy() #skimage.io.imread(labels[rand_index])


			# Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired
			dimz, dimy, dimx = image_u.shape

			# The same for pair
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			image_p = image_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			membr_p = membr_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			label_p = label_p[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]


			# Randomly for unpaired for pair
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			image_u = image_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			membr_u = membr_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			label_u = label_u[randz:randz+DIMZ,randy:randy+DIMY,randx:randx+DIMX]


			seed = np.random.randint(0, 20152015)
			seed_image = np.random.randint(0, 2015)
			seed_membr = np.random.randint(0, 2015)
			seed_label = np.random.randint(0, 2015)

			if self.isTrain:
				# Augment the pair image for same seed
				image_p = self.random_flip(image_p, seed=seed)        
				image_p = self.random_reverse(image_p, seed=seed)
				image_p = self.random_square_rotate(image_p, seed=seed)           
				image_p = self.random_elastic(image_p, seed=seed)

				membr_p = self.random_flip(membr_p, seed=seed)        
				membr_p = self.random_reverse(membr_p, seed=seed)
				membr_p = self.random_square_rotate(membr_p, seed=seed)   
				membr_p = self.random_elastic(membr_p, seed=seed)

				label_p = self.random_flip(label_p, seed=seed)        
				label_p = self.random_reverse(label_p, seed=seed)
				label_p = self.random_square_rotate(label_p, seed=seed)   
				label_p = self.random_elastic(label_p, seed=seed)
				
				# Augment the unpair image for different seed seed
				image_u = self.random_flip(image_u, seed=seed_image)        
				image_u = self.random_reverse(image_u, seed=seed_image)
				image_u = self.random_square_rotate(image_u, seed=seed_image)           
				image_u = self.random_elastic(image_u, seed=seed_image)

				membr_u = self.random_flip(membr_u, seed=seed_membr)        
				membr_u = self.random_reverse(membr_u, seed=seed_membr)
				membr_u = self.random_square_rotate(membr_u, seed=seed_membr)   
				membr_u = self.random_elastic(membr_u, seed=seed_membr)

				label_u = self.random_flip(label_u, seed=seed_label)        
				label_u = self.random_reverse(label_u, seed=seed_label)
				label_u = self.random_square_rotate(label_u, seed=seed_label)   
				label_u = self.random_elastic(label_u, seed=seed_label)


			# Calculate membrane
			def membrane(label):
				membr = np.zeros_like(label)
				for z in range(membr.shape[0]):
					membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
				membr = 255*membr
				membr[label==0] = 0 
				return membr

			membr_p = membrane(membr_p.copy())
			membr_u = membrane(membr_u.copy())

			# Calculate linear label
			label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)
			label_u, nb_labels_u = skimage.measure.label(label_u.copy(), return_num=True)

			label_p = label_p.astype(np.float32)
			label_u = label_u.astype(np.float32)

			label_p = np_2tanh(label_p, maxVal=MAX_LABEL)
			label_u = np_2tanh(label_u, maxVal=MAX_LABEL)

			label_p = np_2imag(label_p, maxVal=255.0)
			label_u = np_2imag(label_u, maxVal=255.0)

			#Expand dim to make single channel
			image_p = np.expand_dims(image_p, axis=-1)
			membr_p = np.expand_dims(membr_p, axis=-1)
			label_p = np.expand_dims(label_p, axis=-1)

			image_u = np.expand_dims(image_u, axis=-1)
			membr_u = np.expand_dims(membr_u, axis=-1)
			label_u = np.expand_dims(label_u, axis=-1)

			yield [image_p.astype(np.float32), 
				   membr_p.astype(np.float32), 
				   label_p.astype(np.float32), 
				   image_u.astype(np.float32), 
				   membr_u.astype(np.float32), 
				   label_u.astype(np.float32),
				   ] 

	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_reverse = np.random.randint(1,3)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
				
	def random_elastic(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		old_shape = image.shape

		if image.ndim==2:
			image = np.expand_dims(image, axis=0) # Make 3D
		new_shape = image.shape
		dimx, dimy = new_shape[1], new_shape[2]
		size = np.random.randint(4,16) #4,32
		ampl = np.random.randint(2, 5) #4,8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation    import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
		DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
		X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(new_shape[0]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[z,...])
			flowZ  = map_coordinates(imageZ, indices, order=0).astype(np.float32)

			warpedZ = flowZ.reshape(image[z,...].shape)
			warped[z,...] = warpedZ     
		warped = np.reshape(warped, old_shape)
		return warped


class Model(GANModelDesc):
	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img):
		assert img is not None
		return arch_generator(img)
		# return arch_fusionnet(img)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)


	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'membr_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label_p'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'image_u'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'membr_u'),
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 1), 'label_u'),
			]
	def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
		with tf.name_scope(name=name):
			# the Wasserstein-GAN losses
			d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
			g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
			# add_moving_summary(self.d_loss, self.g_loss)
			return g_loss, d_loss

	def _build_graph(self, inputs):
		G = tf.get_default_graph() # For round
		tf.local_variables_initializer()
		tf.global_variables_initializer()
		pi, pm, pl, ui, um, ul = inputs
		pi = cvt2tanh(pi)
		pm = cvt2tanh(pm)
		pl = cvt2tanh(pl)
		ui = cvt2tanh(ui)
		um = cvt2tanh(um)
		ul = cvt2tanh(ul)


		# def tf_membr(label):
		# 	with freeze_variables():
		# 		label = np_2imag(label, maxVal=MAX_LABEL)
		# 		label = np.squeeze(label) # Unimplemented: exceptions.NotImplementedError: Only for images of dimension 1-3 are supported, got a 4D one
		# 		# label, nb_labels = skimage.measure.label(color, return_num=True)
		# 		# label = np.expand_dims(label, axis=-1).astype(np.float32) # Modify here for batch
		# 		# for z in range(membr.shape[0]):
		# 		# 	membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
		# 		membr = 1-skimage.segmentation.find_boundaries(np.squeeze(label), mode='thick') #, mode='inner'
		# 		membr = np.expand_dims(membr, axis=-1).astype(np.float32)
		# 		membr = np.expand_dims(membr, axis=0).astype(np.float32)
		# 		membr = np_2tanh(membr, maxVal=1.0)
		# 		membr = np.reshape(membr, label.shape)
		# 		return membr
		
		# def tf_label(color):
		# 	with freeze_variables():
		# 		color = np_2imag(color, maxVal=MAX_LABEL)
		# 		color = np.squeeze(color) # Unimplemented: exceptions.NotImplementedError: Only for images of dimension 1-3 are supported, got a 4D one
		# 		label, nb_labels = skimage.measure.label(color, return_num=True)
		# 		label = np.expand_dims(label, axis=-1).astype(np.float32)
		# 		label = np.expand_dims(label, axis=0).astype(np.float32)
		# 		label = np_2tanh(label, maxVal=MAX_LABEL)
		# 		label = np.reshape(label, color.shape)
		# 		return label

		def tf_rand_score (x1, x2):
			return 1.0 - adjusted_rand_score (x1.flatten (), x2.flatten ())

		def rounded(label, factor = MAX_LABEL, name='quantized'):
			with G.gradient_override_map({"Round": "Identity"}):
				with freeze_variables():
					with tf.name_scope(name=name):
						label = cvt2imag(label, maxVal=factor)
						label = tf.round(label)
						label = cvt2tanh(label, maxVal=factor)
					return tf.identity(label, name=name)


		with argscope([Conv2D, Deconv2D, FullyConnected],
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NHWC'), \
				argscope(LeakyReLU, alpha=0.2):

			

			with tf.variable_scope('gen'):
				# Real pair image 4 gen
				with tf.variable_scope('I2M'):
					pim = self.generator(pi)
				with tf.variable_scope('M2L'):
					piml  = self.generator(pim)
					pml   = self.generator(pm)
					# piml  = tf.py_func(tf_label, [(pim)], tf.float32)
					# pml   = tf.py_func(tf_label, [(pm)], tf.float32)
					# print pim
					# print piml
				# with tf.variable_scope('L2M'):
				# # with freeze_variables():
				# 	pimlm = self.generator(piml) #
				# 	plm   = self.generator(pl)
				# 	pmlm  = self.generator(pml)		
				# 	# pimlm = tf.py_func(tf_membr, [(piml)], tf.float32) #
				# 	# plm   = tf.py_func(tf_membr, [(pl)	], tf.float32)
				# 	# pmlm  = tf.py_func(tf_membr, [(pml)	], tf.float32)
				# 	# print piml
				# 	# print pimlm
				# with tf.variable_scope('M2I'):
				# 	pimlmi = self.generator(pimlm) #
				# 	pimi   = self.generator(pim)

				# # Real pair label 4 gen
				# with tf.variable_scope('L2M'):
				# # with freeze_variables():
				# 	plm = self.generator(pl)
				# 	# plm  = tf.py_func(tf_membr, [(pl)	, tf.float32])
				# with tf.variable_scope('M2I'):
				# 	plmi = self.generator(plm)
				# 	pmi  = self.generator(pi)
				# with tf.variable_scope('I2M'):
				# 	plmim = self.generator(plmi) #
				# 	pim   = self.generator(pi)
				# 	pmim  = self.generator(pmi)

				# with tf.variable_scope('M2L'):
				# 	plmiml = self.generator(plmim) #
				# 	plml   = self.generator(plm)
				# 	# plmiml = tf.py_func(tf_label, [(plmim)], tf.float32)
				# 	# plml   = tf.py_func(tf_label, [(plm)], tf.float32)

			with tf.variable_scope('discrim'):
				# with tf.variable_scope('I'):
				# 	i_dis_real 			  = self.discriminator(ui)
				# 	i_dis_fake_from_label = self.discriminator(plmi)
				with tf.variable_scope('M'):
					m_dis_real 			  = self.discriminator(um)
					m_dis_fake_from_image = self.discriminator(pim)
					# m_dis_fake_from_label = self.discriminator(plm)
				with tf.variable_scope('L'):
					l_dis_real 			  = self.discriminator(ul)
					l_dis_fake_from_image = self.discriminator(piml)
		


		piml  = rounded(piml) #
		pml   = rounded(pml)
		# plmiml = rounded(plmiml) #
		# plml   = rounded(plml)


		# with tf.name_scope('Recon_I_loss'):
		# 	recon_imi 		= tf.reduce_mean(tf.abs((pi) - (pimi)), name='recon_imi')
		# 	recon_lmi 		= tf.reduce_mean(tf.abs((pi) - (plmi)), name='recon_lmi')
		# 	recon_imlmi 	= tf.reduce_mean(tf.abs((pi) - (pimlmi)), name='recon_imlmi') #

		with tf.name_scope('Recon_L_loss'):
			# recon_lml 		= tf.reduce_mean(tf.abs((pl) - (plml)), name='recon_lml')
			recon_iml 		= tf.reduce_mean(tf.abs((pl) - (piml)), name='recon_iml')
			# recon_lmiml 	= tf.reduce_mean(tf.abs((pl) - (plmiml)), name='recon_lmiml') #

		with tf.name_scope('Recon_M_loss'):
			# recon_mim 		= tf.reduce_mean(tf.abs((pm) - (pmim)), name='recon_mim')
			# recon_mlm 		= tf.reduce_mean(tf.abs((pm) - (pmlm)), name='recon_mlm')

			recon_im 		= tf.reduce_mean(tf.abs((pm) - (pim)), name='recon_im')
			# recon_lm 		= tf.reduce_mean(tf.abs((pm) - (plm)), name='recon_lm')
			
		with tf.name_scope('GAN_loss'):
			# G_loss_IL, D_loss_IL = self.build_losses(i_dis_real, i_dis_fake_from_label, name='IL')
			G_loss_LI, D_loss_LI = self.build_losses(l_dis_real, l_dis_fake_from_image, name='LL')
			G_loss_MI, D_loss_MI = self.build_losses(m_dis_real, m_dis_fake_from_image, name='MI')
			# G_loss_ML, D_loss_ML = self.build_losses(m_dis_real, m_dis_fake_from_label, name='ML')

		# custom loss for membr
		with tf.name_scope('membr_loss'):
			def membr_loss(y_true, y_pred, name='membr_loss'):
				return tf.reduce_mean(tf.subtract(binary_cross_entropy(cvt2imag(y_true, maxVal=1.0), cvt2imag(y_pred, maxVal=1.0)), 
								   dice_coe(cvt2imag(y_true, maxVal=1.0), cvt2imag(y_pred, maxVal=1.0), axis=[1,2,3], loss_type='jaccard')),  name=name)
			membr_im = membr_loss(pm, pim, name='membr_im')
			# print membr_im
			# membr_lm = membr_loss(pm, plm, name='membr_lm')
			# membr_imlm = membr_loss(pm, pimlm, name='membr_imlm')
			# membr_lmim = membr_loss(pm, plmim, name='membr_lmim')
			# membr_mlm = membr_loss(pm, pmlm, name='membr_mlm')
			# membr_mim = membr_loss(pm, pmim, name='membr_mim')
		# custom loss for label
		with tf.name_scope('label_loss'):
			def label_loss(y_true_L, y_pred_L, y_grad_M, name='label_loss'):
				g_mag_grad_M = cvt2imag(y_grad_M, maxVal=1.0)
				mag_grad_L   = magnitute_central_difference(y_pred_L, name='mag_grad_L')
				cond = tf.greater(mag_grad_L, tf.zeros_like(mag_grad_L))
				thresholded_mag_grad_L = tf.where(cond, 
										   tf.ones_like(mag_grad_L), 
										   tf.zeros_like(mag_grad_L), 
										   name='thresholded_mag_grad_L')

				gtv_guess = tf.multiply(g_mag_grad_M, thresholded_mag_grad_L, name='gtv_guess')
				loss_gtv_guess = tf.reduce_mean(gtv_guess, name='loss_gtv_guess')

				thresholded_mag_grad_L = cvt2tanh(thresholded_mag_grad_L, maxVal=1.0)
				gtv_guess = cvt2tanh(gtv_guess, maxVal=1.0)
				return loss_gtv_guess, thresholded_mag_grad_L

			label_iml, g_iml = label_loss(None, piml, pim, name='label_iml')
			# label_lml, g_lml = label_loss(None, plml, plm, name='label_lml')
			# label_lmiml, g_lmiml = label_loss(None, plmiml, plmim, name='label_lmiml')
			label_ml,  g_ml  = label_loss(None, pml,  pm,  name='label_loss_ml')

		# custom loss for tf_rand_score
		with tf.name_scope('rand_loss'):
			rand_iml = tf.reduce_mean(tf.cast(tf.py_func (tf_rand_score, [piml, pl], tf.float64), tf.float32))
			rand_ml  = tf.reduce_mean(tf.cast(tf.py_func (tf_rand_score, [pml,  pl], tf.float64), tf.float32))


		self.g_loss = tf.add_n([
								#(recon_imi), # + recon_lmi + recon_imlmi), #
								(recon_iml), # + recon_lml + recon_lmiml), #
								(recon_im), #  + recon_lm + recon_mim + recon_mlm),
								(rand_iml), # + rand_lml + rand_lmiml), #
								(rand_ml), #  + rand_lm + rand_mim + rand_mlm),
								# (G_loss_IL + G_loss_LI + G_loss_MI + G_loss_ML), 
								(G_loss_LI + G_loss_MI), 
								(membr_im), # + membr_lm + membr_imlm + membr_lmim + membr_mlm + membr_mim),
								# (label_iml + label_lml + label_lmiml + label_ml)
								(label_iml + label_ml)
								], name='G_loss_total')
		self.d_loss = tf.add_n([
								# (D_loss_IL + D_loss_LI + D_loss_MI + D_loss_ML), 
								(D_loss_LI + D_loss_MI), 
								], name='D_loss_total')

		wd_g = regularize_cost('gen/.*/W', 		l2_regularizer(1e-5), name='G_regularize')
		wd_d = regularize_cost('discrim/.*/W', 	l2_regularizer(1e-5), name='D_regularize')

		self.g_loss = tf.add(self.g_loss, wd_g, name='g_loss')
		self.d_loss = tf.add(self.d_loss, wd_d, name='d_loss')

	

		self.collect_variables()

		add_moving_summary(self.d_loss, self.g_loss)
		add_moving_summary(
			recon_iml, 
			recon_im, 
			label_iml, 
			label_ml, 
			# rand_iml, 
			# rand_ml, 
			# membr_im
			# recon_imi, recon_lmi, recon_imlmi,
			# recon_lml, recon_iml, recon_lmiml,
			# recon_mim, recon_mlm, recon_im , recon_lm,
			)


		viz = tf.concat([tf.concat([ui, pi, pim, piml, g_iml], 2), 
						 # tf.concat([ul, pl, plm, plmi, plmim, plmiml], 2),
						 tf.concat([um, pl, pm, pml, g_ml], 2),
						 # tf.concat([pl, pl, g_iml, g_lml, g_lmiml,   g_ml], 2),
						 ], 1)
		# add_moving_summary(
		# 	recon_imi, recon_lmi,# recon_imlmi,
		# 	recon_lml, recon_iml,# recon_lmiml,
		# 	recon_mim, recon_mlm, recon_im , recon_lm,
		# 	)
		# viz = tf.concat([tf.concat([ui, pi, pim, piml], 2), 
		# 				 tf.concat([ul, pl, plm, plmi], 2),
		# 				 tf.concat([um, pm, pmi, pmim], 2),
		# 				 tf.concat([um, pm, pml, pmlm], 2),
		# 				 ], 1)
		viz = cvt2imag(viz)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)

	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
###############################################################################
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image_p', 'membr_p', 'label_p', 'image_u', 'membr_u', 'label_u'], ['viz'])

	def _before_train(self):
		global args
		self.test_ds = get_data(args.data, isTrain=False, isValid=False, isTest=True)

	def _trigger(self):
		for lst in self.test_ds.get_data():
			viz_test = self.pred(lst)
			viz_test = np.squeeze(np.array(viz_test))

			#print viz_test.shape

			self.trainer.monitors.put_image('viz_test', viz_test)
###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
	# Process the directories 
	if isTrain:
		num=500
		names = ['trainA', 'trainB']
	if isValid:
		num=1
		names = ['trainA', 'trainB']
	if isTest:
		num=1
		names = ['testA', 'testB']

	
	dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
						  os.path.join(dataDir, names[1]),
						  num, 
						  isTrain=isTrain, 
						  isValid=isValid, 
						  isTest =isTest)
	return dset
###############################################################################
class ClipCallback(Callback):
	def _setup_graph(self):
		vars = tf.trainable_variables()
		ops = []
		for v in vars:
			n = v.op.name
			if not n.startswith('discrim/'):
				continue
			logger.info("Clip {}".format(n))
			ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
		self._op = tf.group(*ops, name='clip')

	def _trigger_step(self):
		self._op.run()
###############################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',   required=True, 
									help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load',   help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()
	# python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

	
	train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
	# valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)
	# test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)

	# train_ds = PrintData(train_ds)
	# valid_ds = PrintData(valid_ds)
	# test_ds  = PrintData(test_ds)
	# Augmentation is here
	

	# data_set  = ConcatData([train_ds, valid_ds])
	data_set  = train_ds
	# data_set  = LocallyShuffleData(data_set, buffer_size=4)
	# data_set  = AugmentImageComponent(data_set, augmentors, (0)) # Only apply for the image


	data_set  = PrintData(data_set)
	data_set  = PrefetchDataZMQ(data_set, 5)
	data_set  = QueueInput(data_set)
	model 	  = Model()

	os.environ['PYTHONWARNINGS'] = 'ignore'

	# Set the GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Running train or deploy
	if args.sample:
		# TODO
		# sample
		pass
	else:
		# Set up configuration
		# Set the logger directory
		logger.auto_set_dir()

		# SyncMultiGPUTrainer(config).train()
		nr_tower = max(get_nr_gpu(), 1)
		if nr_tower == 1:
			trainer = SeparateGANTrainer(data_set, model, g_period=4, d_period=1)
		else:
			trainer = MultiGPUGANTrainer(nr_tower, data_set, model)
		trainer.train_with_defaults(
			callbacks=[
				# PeriodicTrigger(ModelSaver(), every_k_epochs=20),
				ClipCallback(),
				ScheduledHyperParamSetter('learning_rate', 
					[(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				],
			session_init=SaverRestore(args.load) if args.load else None, 
			steps_per_epoch=data_set.size(),
			max_epoch=300, 
		)