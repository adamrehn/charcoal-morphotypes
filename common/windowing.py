import cv2, functools, glob, keras, random, maskutils, math, os
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
import slidingwindow as sw
import numpy as np

# From <https://github.com/keras-team/keras-preprocessing/blob/1.0.2/keras_preprocessing/image.py#L341>
# Duplicated here since this is not exposed by all versions of Keras
def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x


class WindowSequenceBase(keras.utils.Sequence):
	'''
	Base Sequence class for sets of images to which a sliding window is applied.
	'''
	
	def __init__(self, directory, batchSize, windowSize, validationSplit, subset, numLabels):
		
		# Store our parameters
		self.batchSize = batchSize
		self.windowSize = windowSize
		self.subset = subset
		self.numLabels = numLabels
		
		# Validate the specified subset string
		if self.subset not in ['training', 'validation']:
			raise RuntimeError('invalid subset "{}"'.format(self.subset))
		
		# Retrieve the list of input images in the input directory
		images = sorted(glob.glob(os.path.join(directory, '*.png')))
		totalImages = len(images)
		
		# Slice the list of input images based on our subset
		validationOffset = math.floor((1.0 - validationSplit) * totalImages)
		self.images = images[validationOffset:] if subset == 'validation' else images[:validationOffset]
	
	@functools.lru_cache(maxsize=5)
	def _getDataAndLabels(self, imagePath):
		'''
		Retrieves the image data and associated classification label(s) for the specified image file.
		The return value is a tuple containing the image data and the per-pixel labels, both as NumPy arrays.
		'''
		raster = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
		channels, mask = maskutils.splitAlphaMask(raster)
		ret, mask = cv2.threshold(mask, self.numLabels-1, self.numLabels-1, cv2.THRESH_BINARY)
		return channels, mask
	
	def computeWeightings(self, numImages = 10):
		'''
		Computes the label weightings to use for training, based on the extent to which each classification
		label is represented in a random sample of the training data
		'''
		
		# Sample the masks from up to the user-specified number of randomly-selected images
		samples = random.sample(self.images, min(len(self.images), numImages))
		totals = np.zeros((self.numLabels), dtype=np.int64)
		for sample in samples:
			channels, mask = self._getDataAndLabels(sample)
			histogram, bins = np.histogram(mask, bins = list(range(0, self.numLabels+1)))
			totals += histogram
		
		# Compute the label weightings
		overall = np.sum(totals)
		weights = 1.0 - (totals / overall)
		return weights


class PreWindowedSequence(WindowSequenceBase):
	'''
	Sequence class that reads pre-windowed images from an input directory.
	Images should have per-pixel classification labels and all be of the same resolution.
	'''
	
	def __init__(self, directory, batchSize, windowSize, validationSplit, subset, numLabels):
		'''
		Creates a new sequence for the images in the specified input directory.
		Note that the resolution of all of the images in the input directory must match the specified window size.
		
		Arguments:
		- `directory`: the input directory containing the images
		- `batchSize`: the batch size
		- `windowSize`: the size of the sliding window, in pixels
		- `validationSplit`: the percentage of input images to be used as validation data
		- `subset`: which subset of images to iterate over, either 'training' or 'validation'
		'''
		
		# Validate and store our parameters
		super().__init__(directory, batchSize, windowSize, validationSplit, subset, numLabels)
		
		# Verify that the dimensions of the first input image match the specified window size
		firstImage = image.img_to_array(image.load_img(self.images[0], color_mode='grayscale'))
		if firstImage.shape[0] != self.windowSize or firstImage.shape[1] != self.windowSize:
			raise RuntimeError('all images must have a resolution of {}x{}, found an image of {}x{}'.format(
				self.windowSize,
				self.windowSize,
				firstImage.shape[1],
				firstImage.shape[0]
			))
		
		# Display the number of detected images
		print('Found {} pre-windowed {} image(s) with resolution {}x{}'.format(
			len(self.images),
			subset,
			firstImage.shape[1],
			firstImage.shape[0],
		))
	
	def __len__(self):
		'''
		Returns the length of the iterator in batches
		'''
		return int(math.ceil(len(self.images) / self.batchSize))
	
	def __getitem__(self, index):
		'''
		Returns the batch at the specified index
		'''
		
		# Determine which images have been requested
		startIndex = index * self.batchSize
		endIndex = startIndex + self.batchSize
		
		# Retrieve and transform the image data and labels for the requested windows
		transformed = [self._transformImage(imageIndex) for imageIndex in range(startIndex, endIndex)]
		
		# Split the data from the labels
		data = [result[0] for result in transformed]
		labels = [result[1] for result in transformed]
		
		# Return a tuple containing the data and the labels
		return np.concatenate(data), np.concatenate(labels)
	
	def _transformImage(self, imageIndex):
		'''
		Retrieves the specified image and transforms it as appropriate
		'''
		
		# Retrieve the image data and label(s)
		data, labels = self._getDataAndLabels(self.images[imageIndex % len(self.images)])
		
		# Apply our random transforms if we are processing training data
		if self.subset == 'training':
			
			# Randomly apply a horizontal flip
			if np.random.random() < 0.5:
				data = flip_axis(data, 1)
				labels = flip_axis(labels, 1)
			
			# Randomly apply a vertical flip
			if np.random.random() < 0.5:
				data = flip_axis(data, 0)
				labels = flip_axis(labels, 0)
		
		# One-hot encode the labels
		labels = np.expand_dims(to_categorical(np.ravel(labels), self.numLabels), axis=0)
		
		return np.expand_dims(data, axis=0), labels


class SlidingWindowSequence(WindowSequenceBase):
	'''
	Sequence class that runs a sliding window over the images in an input directory.
	Images should have per-pixel classification labels and all be of the same resolution.
	'''
	
	def __init__(self, directory, batchSize, windowSize, overlap, validationSplit, subset, numLabels):
		'''
		Creates a new sequence for the images in the specified input directory.
		Note that all of the images in the input directory must be of the same resolution.
		
		Arguments:
		- `directory`: the input directory containing the images
		- `batchSize`: the batch size
		- `windowSize`: the size of the sliding window, in pixels
		- `overlap`: the percentage of overlap between the generated windows
		- `validationSplit`: the percentage of input images to be used as validation data
		- `subset`: which subset of images to iterate over, either 'training' or 'validation'
		'''
		
		# Validate and store our parameters
		super().__init__(directory, batchSize, windowSize, validationSplit, subset, numLabels)
		self.overlap = overlap
		
		# Use the dimensions of the first input image to determine how many windows each image will contain
		firstImage = image.img_to_array(image.load_img(self.images[0], color_mode='grayscale'))
		windows = self._generateWindows(firstImage)
		self.windowsPerImage = len(windows)
		
		# Display the number of detected images
		print('Found {} {} image(s) with resolution {}x{} ({} windows per image)'.format(
			len(self.images),
			subset,
			firstImage.shape[1],
			firstImage.shape[0],
			self.windowsPerImage
		))
	
	def _transformIndex(self, index):
		'''
		Transforms a one-dimensional overall window index into a tuple of (image,window)
		'''
		return index // self.windowsPerImage, index % self.windowsPerImage
	
	def __len__(self):
		'''
		Returns the length of the iterator in batches
		'''
		return int(math.ceil((len(self.images) * self.windowsPerImage) / self.batchSize))
	
	def __getitem__(self, index):
		'''
		Returns the batch at the specified index
		'''
		
		# Transform the supplied index to determine which windows from which images have been requested
		startIndex = index * self.batchSize
		endIndex = startIndex + self.batchSize
		startImage, startWindow = self._transformIndex(startIndex)
		endImage, endWindow = self._transformIndex(endIndex)
		
		# Generate the list of requested images
		images = list(range(startImage, endImage+1))
		if endWindow == 0:
			images = images[:-1]
		
		# Generate the list of required windows for each requested image
		imageWindows = []
		for image in images:
			if image == startImage and image == endImage:
				imageWindows.append(list(range(startWindow, endWindow)))
			elif image == startImage:
				imageWindows.append(list(range(startWindow, self.windowsPerImage)))
			elif image == endImage:
				imageWindows.append(list(range(0, endWindow)))
			else:
				imageWindows.append(list(range(0, self.windowsPerImage)))
		
		# Retrieve and transform the image data and labels for the requested windows
		transformed = [self._transformWindows(image, windows) for image, windows in zip(images, imageWindows)]
		
		# Split the data from the labels
		data = [result[0] for result in transformed]
		labels = [result[1] for result in transformed]
		
		# Return a tuple containing the data and the labels
		return np.concatenate(data), np.concatenate(labels)
	
	def _generateWindows(self, data):
		'''
		Generates the set of sliding windows for the supplied NumPy array
		'''
		return sw.generate(data, sw.DimOrder.HeightWidthChannel, self.windowSize, self.overlap)
	
	def _transformWindows(self, imageIndex, windowIndices):
		'''
		Applies the specified set of windows to the specified image and transforms them as appropriate
		'''
		
		# Retrieve the image data and label(s)
		data, labels = self._getDataAndLabels(self.images[imageIndex % len(self.images)])
		
		# Extract the specified set of windows
		windows = self._generateWindows(data)
		windows = [windows[index] for index in windowIndices]
		
		# Apply our random transforms if we are processing training data
		data = [window.apply(data) for window in windows]
		labels = [window.apply(labels) for window in windows]
		if self.subset == 'training':
			
			# Randomly apply a horizontal flip
			if np.random.random() < 0.5:
				data = [flip_axis(window, 1) for window in data]
				labels = [flip_axis(window, 1) for window in labels]
			
			# Randomly apply a vertical flip
			if np.random.random() < 0.5:
				data = [flip_axis(window, 0) for window in data]
				labels = [flip_axis(window, 0) for window in labels]
		
		# Convert the list of windows into a single NumPy array
		data = np.concatenate([np.expand_dims(d, axis=0) for d in data])
		
		# One-hot encode the labels
		labels = np.concatenate([
			np.expand_dims(to_categorical(np.ravel(label), self.numLabels), axis=0)
			for label in labels
		])
		
		return data, labels
