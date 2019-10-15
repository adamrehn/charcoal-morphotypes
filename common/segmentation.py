from segnet import FastSegNet as SegNet, MaxPoolingWithArgmax2D, MaxUnpooling2D
import cv2, keras, maskutils, slidingwindow, os
from .windowing import PreWindowedSequence
from .configuration import Configuration
from .training import TrainingUtils
from .logging import Logger
import slidingwindow as sw
import numpy as np

# The image resolution (and dimensionality) we use with SegNet
SEGNET_IMAGE_SIZE = 512
SEGNET_IMAGE_DIMS = (SEGNET_IMAGE_SIZE, SEGNET_IMAGE_SIZE, 3)

# The learning rate for training
LEARNING_RATE = 1e-4

# The percentage of training data to use as validation data (0.0 to 1.0)
VALIDATION_SPLIT = 0.1

# The number of epochs to train for
EPOCHS = 1000

# The batch size used for training and inference
BATCH_SIZE = 2

class SegmentationNetwork:
	
	@staticmethod
	def denoise(mask):
		'''
		Removes noise in a binary classification mask, reducing the likelihood of incorrect slicing results
		'''
		
		# Perform a median filter to remove salt-and-pepper noise
		mask = cv2.medianBlur(mask, 7)
		
		# Perform closing (dilation + erosion) to remove any holes inside individual charcoal particles
		kernel = np.ones((5,5),np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		
		# Perform opening (erosion + dilation) to remove any loose noise outside individual charcoal particles
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		
		return mask
	
	
	@staticmethod
	def sliceExact(channels, mask, minImageSize = 82):
		'''
		Slices an input image into its individual components using its binary mask channel and yields the results.
		
		Note that this method returns the rotated rectange specifying the exact bounds of each identified particle.
		For a version that returns square bounds suitable for extracting raster subsets, see `SegmentationNetwork.slice()`.
		
		The `minImageSize` argument determines the size threshold below which components are considered to be noise and discarded.
		The default value of 82x82px is based on DSLR microscope images where 1.3px equals approximately 1 micron (13px == 10 microns).
		'''
		
		# Denoise the mask
		mask = SegmentationNetwork.denoise(mask)
		
		# Use the "connected components" algorithm to extract the rotated rectangles for each identified instance of a charcoal particle
		instancesForValues = maskutils.extractConnectedComponents(channels, mask, ignoreZero=True, preciseBounds=True)
		
		# Extract each identified instance
		for maskValue, instances in instancesForValues.items():
			for instanceNum, instance in enumerate(instances):
				
				# If the instance falls below our size threshold, it is likely not a full particle
				centre, size, rotation = instance
				if size[0] < minImageSize or size[1] < minImageSize:
					continue
				
				# Extract the instance 
				yield instance
	
	
	@staticmethod
	def slice(channels, mask, minImageSize = 32):
		'''
		Slices an input image into its individual components using its binary mask channel and yields the results
		
		The `minImageSize` argument determines the size threshold below which components are considered to be noise and discarded.
		The default value of 32x32px is based on DSLR microscope images where 1px equals approximately 1 micron.
		'''
		
		# Denoise the mask
		mask = SegmentationNetwork.denoise(mask)
		
		# Use the "connected components" algorithm to extract the rectangles for each identified instance of a charcoal particle
		instancesForValues = maskutils.extractConnectedComponents(channels, mask, ignoreZero=True)
		
		# Extract each identified instance
		for maskValue, instances in instancesForValues.items():
			for instanceNum, instance in enumerate(instances):
				
				# If the instance falls below our size threshold, it is likely not a full particle
				x,y,w,h = instance
				if w < minImageSize and h < minImageSize:
					continue
				
				# Pad/crop the bounding box so the image is square, and add 10px extra padding
				squareSize = max(w, h)
				x,y,w,h = slidingwindow.fitToSize((x,y,w,h), squareSize, squareSize, channels.shape)
				x,y,w,h = slidingwindow.padRectEqually((x,y,w,h), 10, channels.shape)
				
				# Extract the instance 
				yield (x,y,w,h), channels[ y:y+h, x:x+w ], mask[ y:y+h, x:x+w ]
	
	
	@staticmethod
	def sliceToDir(infile, outdir, minImageSize = 32, includeMask = False, warnOnOverwrite = False):
		'''
		Slices an input file into its individual components using its binary mask channel and saves the results to disk
		'''
		
		# Determine the basename of the input file without any extension
		imageName = os.path.splitext(os.path.basename(infile))[0]
		
		# Split the binary mask from the other image channels
		raster = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
		channels, mask = maskutils.splitAlphaMask(raster)
		
		# Save each identified instance as a separate image
		for instanceNum, instance in enumerate(SegmentationNetwork.slice(channels, mask, minImageSize)):
			
			# Determine if we are including the mask in the output file
			instanceRect, instanceChannels, instanceMask = instance
			if includeMask == True:
				combined = np.zeros((instanceChannels.shape[0], instanceChannels.shape[1], instanceChannels.shape[2]+1), dtype=instanceChannels.dtype)
				combined[:,:,0:instanceChannels.shape[2]] = instanceChannels
				combined[:,:,instanceChannels.shape[2]] = instanceMask
				instanceChannels = combined
			
			# If the file already exists and warnings have been requested, emit a warning
			outfile = os.path.join(outdir, '{}-{}.png'.format(imageName, instanceNum))
			if warnOnOverwrite == True and os.path.exists(outfile):
				Logger.warning('overwriting existing file {}'.format(outfile))
			
			# Save the instance to file
			cv2.imwrite(outfile, instanceChannels)
	
	
	@staticmethod
	def windowToDir(infile, outdir, warnOnOverwrite = False):
		'''
		Applies a sliding window to an input file so that each window is the correct size for use as
		input for the segmentation network, and saves the results to disk
		'''
		
		# Determine the basename of the input file without any extension
		imageName = os.path.splitext(os.path.basename(infile))[0]
		
		# Read the image data
		raster = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
		channels, mask = maskutils.splitAlphaMask(raster)
		
		# Run a sliding window over the data and save each window as a separate image
		windows = sw.generate(raster, sw.DimOrder.HeightWidthChannel, SEGNET_IMAGE_SIZE, 0.5)
		for windowNum, window in enumerate(windows):
			
			# If the file already exists and warnings have been requested, emit a warning
			outfile = os.path.join(outdir, '{}-window{}.png'.format(imageName, windowNum))
			if warnOnOverwrite == True and os.path.exists(outfile):
				Logger.warning('overwriting existing file {}'.format(outfile))
			
			# If the window is smaller than required (due to the input image being smaller), discard it
			if window.w < SEGNET_IMAGE_SIZE or window.h < SEGNET_IMAGE_SIZE:
				Logger.warning('discarding window of size {}x{} because it is too small'.format(window.w, window.h))
				continue
			
			# If the window does not contain any pixels representing a charcoal particle, discard it
			if np.count_nonzero(window.apply(mask)) == 0:
				Logger.warning('discarding window because it does not contain any particle pixels')
				continue
			
			# Save the instance to file
			cv2.imwrite(outfile, window.apply(raster))
	
	
	@staticmethod
	def particleBounds(mask):
		'''
		Determines the exact bounds of an individual charcoal particle based on its binary classification mask.
		
		Returns a rotated rectangle represented by a tuple of (centre, size, rotation)
		'''
		contiguous = np.ascontiguousarray(mask, dtype=np.uint8)
		_, contours, hierarchy = cv2.findContours(contiguous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		return cv2.minAreaRect(contours[0])
	
	
	@staticmethod
	def particleSize(mask):
		'''
		Determines the size (in pixels) of an individual charcoal particle based on its binary classification mask
		'''
		centre, size, rotation = SegmentationNetwork.particleBounds(mask)
		return size
	
	
	@staticmethod
	def particleElongationRatio(size):
		'''
		Computes the elongation ratio (longer dimension / shorter dimension) of an individual charcoal particle
		'''
		longer = np.max(size)
		shorter = np.min(size)
		return longer / shorter
	
	
	@staticmethod
	def create():
		'''
		Creates the segmentation neural network
		'''
		return SegNet(SEGNET_IMAGE_DIMS, 2)
	
	
	@staticmethod
	def load():
		'''
		Loads the segmentation neural network from the last saved training checkpoint
		'''
		return TrainingUtils.loadCheckpoint('segmentation', {
			'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
			'MaxUnpooling2D': MaxUnpooling2D
		})
	
	
	@staticmethod
	def infer(model, metadata, data, progressCallback = None):
		'''
		Performs inference on the supplied input data
		
		Returns a NumPy array containing the raw predictions (per-pixel probabilities)
		'''
		infer = lambda raster, windows: model.predict(np.array([window.apply(raster) for window in windows])).reshape((len(windows), SEGNET_IMAGE_SIZE, SEGNET_IMAGE_SIZE, 2))
		return sw.mergeWindows(data, sw.DimOrder.HeightWidthChannel, SEGNET_IMAGE_SIZE, 0.5, BATCH_SIZE, infer, progressCallback)
	
	
	@staticmethod
	def loadAndInfer(data, progressCallback = None):
		'''
		Loads the segmentation neural network from the last saved training checkpoint and performs inference on the supplied input data
		
		Returns a NumPy array containing the raw predictions (per-pixel probabilities)
		'''
		
		# Load the model from the last saved checkpoint
		model, metadata = SegmentationNetwork.load()
		
		# Perform inference
		return SegmentationNetwork.infer(model, metadata, data, progressCallback)
	
	
	@staticmethod
	def getValidationData():
		'''
		Retrieves the validation data for the segmentation neural network
		'''
		return PreWindowedSequence(
			Configuration.path('segmentation_data_windowed'),
			BATCH_SIZE,
			SEGNET_IMAGE_SIZE,
			VALIDATION_SPLIT,
			'validation',
			2
		)
	
	
	@staticmethod
	def train():
		'''
		Creates and trains the segmentation neural network
		'''
		
		# Load our training images from the data directory, performing data augmentation
		trainingData = PreWindowedSequence(
			Configuration.path('segmentation_data_windowed'),
			BATCH_SIZE,
			SEGNET_IMAGE_SIZE,
			VALIDATION_SPLIT,
			'training',
			2
		)
		
		# Load our validation images from the data directory
		validationData = SegmentationNetwork.getValidationData()
		
		# Create the SegNet model
		model = SegmentationNetwork.create()
		
		# Compile the model with our desired optimisation algorithm and learning rate
		model.compile(
			loss = 'categorical_crossentropy',
			optimizer = keras.optimizers.RMSprop(),
			metrics = ['categorical_accuracy']
		)
		
		# Compute the weights for each of our outputs, based on the number of training samples present for each class
		print('Computing class weights, this may take some time...')
		classWeights = trainingData.computeWeightings(len(trainingData))
		print('Class weights: {}'.format(classWeights))
		
		# Train the model, saving the best checkpoint to disk
		TrainingUtils.trainWithCheckpoints(
			model,
			'segmentation',
			{},
			trainingData,
			validationData,
			BATCH_SIZE,
			EPOCHS,
			'val_categorical_accuracy',
			100,
			classWeights
		)
