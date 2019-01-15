# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from scipy.linalg import logm, expm

from niftynet.layer.base_layer import RandomisedLayer


class RandomArtefactLayer(RandomisedLayer):
    """
    generate randomised movement artefact for data augmentation
    """

    def __init__(self, name='random_artefact'):
        super(RandomArtefactLayer, self).__init__(name=name)
	self._num_events = None
	self._event_lengths = None
	self._event_times = None
	self._angles = None
	self._trans = None

	self.lambda_large = 2
	self.lambda_small = 20
	self.angles_stddev_large = 0.03*np.pi
	self.angles_stddev_small = 0.001*np.pi
	self.trans_stddev_large = 3
	self.trans_stddev_small = 0.01
	self.acquisition_type = '3D'
	self.interp_mode = 'constant' #'nearest' 
	self.weighting = 'frequency'
	self.instant_events = True
	self.use_itk = False

    def get3DRotationMatrix(self, angles):
	ax = angles[0]
	ay = angles[1]
	az = angles[2]
	Rx = np.array([[1,          0,           0],
      		       [0, np.cos(ax), -np.sin(ax)],
      	               [0, np.sin(ax),  np.cos(ax)]], np.float32)
        Ry = np.array([[np.cos(ay),  0, np.sin(ay)],
      		       [0,           1,          0],
                       [-np.sin(ay), 0, np.cos(ay)]], np.float32)
	Rz = np.array([[np.cos(az), -np.sin(az), 0],
      		       [np.sin(az),  np.cos(az), 0],
      		       [0,                    0, 1]], np.float32)
	R = np.dot(Rz, np.dot(Ry,Rx))
	return R

    def getAffineMatrix(self, R, t, image_3d, method):
	centre = np.array([[0.5*image_3d.shape[0]-0.5, 0.5*image_3d.shape[1]-0.5, 0.5*image_3d.shape[2]-0.5]])
	if method == 0:
		offset = centre - centre.dot(R.T) + t.T
		A = np.concatenate((R, offset.T), axis=1)
		A = np.concatenate((A, np.array([[0,0,0,1]])), axis=0)
		A = np.linalg.inv(A)
		A[3,:] = [0,0,0,1]
	elif method == 1:
		offset = centre - centre.dot(R.T) + t.T
		offset = -offset.dot(R)
		A = np.concatenate((R.T, offset.T), axis=1)
		A = np.concatenate((A, np.array([[0,0,0,1]])), axis=0)
	return A

    def getAffineMatrixITK(self, R, t):
	A = np.concatenate((R, t), axis=1)
	A = np.concatenate((A, np.array([[0,0,0,1]])), axis=0)
	A = np.linalg.inv(A)
	A[3,:] = [0,0,0,1]
	return A

    def affineTranformITK(self, image_3d_itk, A_itk):
	A = sitk.AffineTransform(3)
	t = A_itk[:,3]
	R = A_itk[0:3,0:3]
	A.SetTranslation((t[0],t[1],t[2]))
	A.SetMatrix(R.ravel())
	centre = np.array([0.5*image_3d_itk.GetSize()[0], 0.5*image_3d_itk.GetSize()[1], 0.5*image_3d_itk.GetSize()[2]], np.int)
	A.SetCenter((centre[0],centre[1],centre[2]))
	interpolator = sitk.sitkLinear
	#interpolator = sitk.sitkBSpline
	#interpolator = sitk.sitkCosineWindowedSinc
	ref_img = image_3d_itk
	out_image_itk = sitk.Resample(image_3d_itk, ref_img.GetSize(), A, interpolator, [0,0,0])
	image_out = sitk.GetArrayFromImage(out_image_itk)
	image_out = np.transpose(image_out, (1, 2, 0))
	return image_out

    def computeFourierTransform(self, image_3d):
	F = np.zeros(image_3d.shape, np.float32)
	if self.acquisition_type == '3D':
		F = np.fft.fftshift(np.fft.fftn(image_3d))
	elif self.acquisition_type == '2D':
		for k in xrange(image_3d.shape[2]):
			Ik = image_3d[:,:,k]
			F[:,:,k] = np.fft.fftshift(np.fft.fft2(Ik))
	return F

    def computeInverseFourierTransform(self, F):
	IF = np.zeros(F.shape, np.float32)
	if self.acquisition_type == '3D':
		IF = np.fft.ifftn(np.fft.ifftshift(F))
	elif self.acquisition_type == '2D':
		for k in xrange(F.shape[2]):
			Fk = F[:,:,k]
        		IF[:,:,k] = np.fft.ifft2(np.fft.ifftshift(Fk))
	return IF

    def computeMasks(self, image_3d):
	num_elements = image_3d.size
	event_elements = np.floor(num_elements * self._event_times)
	masks = []
	mask = np.arange(0,event_elements[0,0],1,np.int64)
	masks.append(mask)
	for i in xrange(self._num_events-1):
	    mask = np.arange(event_elements[0,i],event_elements[0,i+1],1,np.int64)
	    masks.append(mask)
	mask = np.arange(event_elements[0,self._num_events-1],num_elements,1,np.int64)
	masks.append(mask)
	return masks

    def computeWeights(self, masks, image_3d):
	num_elements = image_3d.size
	event_elements = np.floor(num_elements * self._event_times)
	crow = np.int(0.5*image_3d.shape[0])
	ccol = np.int(0.5*image_3d.shape[1])
	cdepth = np.int(0.5*image_3d.shape[2])
	if self.weighting == 'time':
	    w1 = np.zeros((1,self._num_events+1), dtype=np.float32)
	    w2 = np.zeros((1,self._num_events+1), dtype=np.float32)
	    w1[0,0:self._num_events] = event_elements
	    w1[0,-1] = num_elements
	    w2[0,1:,] = event_elements[0,:]
	    weights = w1 - w2
	    weights = weights / np.sum(weights)
	elif self.weighting == 'frequency':
	    weights = np.zeros((1,self._num_events+1), dtype=np.float32)
	    xx = np.arange(-ccol,ccol+1,1)
	    yy = np.arange(-crow,crow+1,1)
	    zz = np.arange(-cdepth,cdepth+1,1)
	    X, Y, Z = np.meshgrid(xx,yy,zz)
	    #r = np.divide(1, (np.square(X) + np.square(Y) + np.square(Z) + 1), dtype=np.float32)
	    r = np.exp(-(np.square(X) + np.square(Y) + np.square(Z)), dtype=np.float32)
	    r_t = np.transpose(r, (1, 0, 2))
	    for i in xrange(self._num_events+1):
		mask = masks[i]
		r_masked = r_t[np.unravel_index(mask, image_3d.shape, 'F')]
        	weights[0,i] = np.sum(r_masked) / np.sum(r)
	    weights = weights / np.sum(weights)
	return weights

    def generateRandomEvents(self):
	num_events_large = np.random.poisson(self.lambda_large, 1)[0]
	num_events_small = np.random.poisson(self.lambda_small, 1)[0]
	# If no events, generate one event with p=0.5 of large/small
	if num_events_large == 0 and num_events_small == 0:
	    if np.random.random_sample()>0.5:
        	num_events_large = 1
        	num_events_small = 0
    	    else:
        	num_events_large = 0
        	num_events_small = 1
	return num_events_large, num_events_small

    def randomise(self, spatial_rank=3):
        if spatial_rank == 3:
            self._randomise_artefact_3d()
        else:
            # Currently not supported spatial rank for rand artefact
            pass

    def _randomise_artefact_3d(self):
	if self._num_events is None:
	    # Randomise number of events
	    rand_events = self.generateRandomEvents()
	    num_events_large = rand_events[0]
	    num_events_small = rand_events[1]
	    self._num_events = num_events_large + num_events_small
	    # Randomise angles
	    angles_large = self.angles_stddev_large * np.random.randn(3, num_events_large)
	    angles_small = self.angles_stddev_small * np.random.randn(3, num_events_small)
	    self._angles = np.concatenate((angles_large, angles_small), axis=1)
	    # Randomise translations
	    trans_large = self.trans_stddev_large * np.random.randn(3, num_events_large)
	    trans_small = self.trans_stddev_small * np.random.randn(3, num_events_small)
	    self._trans = np.concatenate((trans_large, trans_small), axis=1)
	    # Randomise event lengths
	    event_lengths_large = np.ceil(np.random.gamma(8, 2, num_events_large))
	    event_lengths_small = np.ceil(np.random.gamma(2, 1, num_events_small))
	    self._event_lengths = np.concatenate((event_lengths_large, event_lengths_small))
	    if self.instant_events == True:
	    	self._event_lengths = np.ones(self._num_events, dtype=np.int)
	    # Randomise times of events
	    self._event_times = np.random.uniform(0, 1, size=(1,self._num_events))
	    #self._event_times = 0.5 + 0.01 * np.random.randn(1,self._num_events)
	    self._event_times = np.sort(self._event_times, axis=1, kind='mergesort')
	    # Shuffle events
	    ids = np.arange(self._num_events)
	    shuffled_ids = np.random.permutation(ids.shape[0])
	    self._event_lengths = self._event_lengths[shuffled_ids]
	    self._angles = self._angles[:,shuffled_ids]
	    self._trans = self._trans[:,shuffled_ids]
        
    def _apply_artefact_3d(self, image_3d, interp_order=3):
        assert image_3d.ndim == 3
	assert self._num_events is not None
	assert self._event_lengths is not None
	assert self._event_times is not None
        assert self._angles is not None
	assert self._trans is not None

	# Transpose input image
	image_3d_t = np.transpose(image_3d, (1, 0, 2))

	# ITK image
	if self.use_itk == True:
	    image_3d_itk = sitk.GetImageFromArray(np.transpose(image_3d, (2, 0, 1)))

	# Fourier transform
	F = self.computeFourierTransform(image_3d)

	# Get 3D rotation matrices
	rotations = []
	for i in xrange(self._num_events):
	     R = self.get3DRotationMatrix(self._angles[:,i])
	     rotations.append(R)

	# Create spatial masks
	masks = self.computeMasks(image_3d)

	# Compute event weights (num_events + 1 for identity transform)
	weights = self.computeWeights(masks, image_3d)
	print('weights:', weights)

	# Create 4x4 affine matrices
	affineMatrices = []
	combinedAffineMatrices = []
	demeanedAffineMatrices = []

	# Id matrix
	R = np.eye(3)
	t = np.array([[0, 0, 0]], np.float32).T
	if self.use_itk == True:
	    Aprev = self.getAffineMatrixITK(R, t)
	else:
	    Aprev = self.getAffineMatrix(R, t, image_3d_t, 0)
	Aavg = weights[0,0] * logm(Aprev)

	# Combine transforms and compute 'average' transform
	for i in xrange(self._num_events):
		R = rotations[i]
		t = np.array([[0, 0, 0]], np.float32).T
		t[0] = self._trans[0,i]
		t[1] = self._trans[1,i]
		t[2] = self._trans[2,i]
		
		if self.use_itk == True:
		    A = self.getAffineMatrixITK(R, t)
		else:
		    A = self.getAffineMatrix(R, t, image_3d_t, 0)
		affineMatrices.append(A)

		combinedA = expm( logm(A) + logm(Aprev) )
		combinedAffineMatrices.append(combinedA)
		Aavg = Aavg + weights[0,i+1]*logm(combinedA)
		Aprev = combinedA
	Aavg = expm(Aavg)
	Ainv = np.linalg.inv(Aavg)
	Ainv[3,:] = [0,0,0,1]

	# De-mean affine transforms
	for i in xrange(self._num_events):
		demeanedA = expm( logm(Ainv) + logm(combinedAffineMatrices[i]) )
		demeanedAffineMatrices.append(demeanedA)

	# Transform 3D data 
	print('Transforming 3D data...')
	F_composite = np.copy(F)
	F_composite_reorder = np.transpose(F_composite, (1, 0, 2))

	# De-mean inital image
	if self.use_itk == True:
	    image_3d_transformed = self.affineTranformITK(image_3d_itk, Ainv)
	else:
	    image_3d_transformed = scipy.ndimage.interpolation.affine_transform(image_3d_t,
										Ainv,
										order=interp_order, 
										mode=self.interp_mode, 
				                                                output_shape=image_3d_t.shape, 
										cval=0.0, 
										prefilter=True, 
										output=np.float32)
	    image_3d_transformed = np.transpose(image_3d_transformed, (1, 0, 2))

	# Fourier transform of de-meaned image
	F_transformed = self.computeFourierTransform(image_3d_transformed)

	# Apply Ainv mask before start of first event
	mask = masks[0]
	Fm = np.copy(F_transformed)
	Fm_reorder = np.transpose(Fm, (1, 0, 2))
	F_composite_reorder[np.unravel_index(mask, F.shape, 'F')] = Fm_reorder[np.unravel_index(mask, F.shape, 'F')]

	# Loop over events
	for i in xrange(self._num_events):
		print(i+1, ' of ', self._num_events)
	    	event_length = self._event_lengths[i]
	    	mask = masks[i+1]
	    
	    	# Get start and end affine transforms
	    	if i == 0:
	    		Astart = np.copy(Ainv)
		else:
			Astart = demeanedAffineMatrices[i-1]
		Aend = demeanedAffineMatrices[i]
	   
	    	for j in xrange(np.int(event_length)):
	    		# Interpolate start and end transforms
			w = (j+1)/float(event_length)
	    		Aj = expm( (1-w)*logm(Astart) + w*logm(Aend) )
			Aj[3,:] = [0,0,0,1] 
		
	    		# Transform image
			if self.use_itk == True:
			    image_3d_transformed = self.affineTranformITK(image_3d_itk, Aj)
			else:
			    image_3d_transformed = scipy.ndimage.interpolation.affine_transform(image_3d_t, 
												Aj,
												order=interp_order,
												mode=self.interp_mode,
												output_shape=image_3d_t.shape,
												cval=0.0,
												prefilter=True,
												output=np.float32)
			    image_3d_transformed = np.transpose(image_3d_transformed, (1, 0, 2))
	 
			# Fourier transform
			F_transformed = self.computeFourierTransform(image_3d_transformed)
	  
	    		# Apply mask at step j
	    		mj = mask[j]
	    		Fm = np.copy(F_transformed)
	    		Fm_reorder = np.transpose(Fm, (1, 0, 2))
			F_composite_reorder[np.unravel_index(mj, F.shape, 'F')] = Fm_reorder[np.unravel_index(mj, F.shape, 'F')]

		# Apply end of mask
	    	mask_end = mask[event_length:,]
	    	Fm = np.copy(F_transformed)
	    	Fm_reorder = np.transpose(Fm, (1, 0, 2))
	    	F_composite_reorder[np.unravel_index(mask_end, F.shape, 'F')] = Fm_reorder[np.unravel_index(mask_end, F.shape, 'F')]
	
	# Transpose back
	F_composite = np.transpose(F_composite_reorder, (1, 0, 2))

	# Inverse Fourier transform
	IF = self.computeInverseFourierTransform(F_composite)
	image_3d = np.abs(IF)	
        return image_3d

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                assert image.shape[-1] == len(interp_orders[field]), \
                    "interpolation orders should be specified for each inputs modality"
                for mod_i, interp_order in enumerate(interp_orders[field]):
                    if image.ndim == 4:
			inputs[field][..., mod_i] = self._apply_artefact_3d(image[..., mod_i], interp_order)
                    elif image.ndim == 5:
                        for t in range(image.shape[-2]):
			    inputs[field][..., t, mod_i] = self._apply_artefact_3d(image[..., t, mod_i], interp_order)
                    else:
                        raise NotImplementedError("unknown input format")

        else:
            raise NotImplementedError("unknown input format")
        return inputs
