from .configuration import Configuration
from .training import TrainingUtils
from keras.applications import vgg16
import numpy as np
import keras, os

# The image resolution required by VGG16
VGG_IMAGE_SIZE = (224, 224)

# The learning rate for training
LEARNING_RATE = 1e-4

# The percentage of training data to use as validation data (0.0 to 1.0)
VALIDATION_SPLIT = 0.1

# The number of epochs to train for
EPOCHS = 1000

# The batch size used for training
BATCH_SIZE = 32

class ClassificationNetwork:
	
	@staticmethod
	def readImage(path):
		'''
		Reads and preprocesses an image file so that it is suitable for consumption by the morphotype classification neural network
		'''
		image = keras.preprocessing.image.load_img(path, target_size=VGG_IMAGE_SIZE)
		array = keras.preprocessing.image.img_to_array(image)
		array = np.expand_dims(array, axis=0)
		return vgg16.preprocess_input(array)
	
	
	@staticmethod
	def create(numClasses):
		'''
		Creates the morphotype classification neural network
		'''
		
		# Instantiate the VGG16 network with ImageNet weights
		vgg = vgg16.VGG16(weights='imagenet')
		
		# Freeze the weights for the first 25 layers
		for layer in vgg.layers[:25]:
			layer.trainable = False
		
		# Replace the final prediction layer with a version that predicts the number of classes present in the training data
		origLayer = vgg.layers[-1]
		newLayer = keras.layers.Dense(numClasses, activation=origLayer.activation, name=origLayer.name)(vgg.layers[-2].get_output_at(0))
		return keras.models.Model(inputs=vgg.get_input_at(0), outputs=newLayer)
	
	
	@staticmethod
	def load():
		'''
		Loads the morphotype classification neural network from the last saved training checkpoint
		'''
		return TrainingUtils.loadCheckpoint('classification')
	
	
	@staticmethod
	def infer(model, metadata, data):
		'''
		Performs inference on the supplied input data
		
		Returns a tuple containing the raw predictions and the list of classification labels
		'''
		return (model.predict(data), metadata)
	
	
	@staticmethod
	def loadAndInfer(data):
		'''
		Loads the morphotype classification neural network from the last saved training checkpoint and performs inference on the supplied input data
		
		Returns a tuple containing the raw predictions and the list of classification labels
		'''
		
		# Load the model from the last saved checkpoint
		model, metadata = ClassificationNetwork.load()
		
		# Perform inference
		return ClassificationNetwork.infer(model, metadata, data)
	
	
	@staticmethod
	def getValidationData():
		'''
		Retrieves the validation data for the morphotype classification neural network
		'''
		genValidation = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, validation_split=VALIDATION_SPLIT)
		return genValidation.flow_from_directory(
			Configuration.path('classification_data'),
			subset = 'validation',
			batch_size = BATCH_SIZE,
			target_size = VGG_IMAGE_SIZE
		)
	
	
	@staticmethod
	def train():
		'''
		Creates and trains the morphotype classification neural network
		'''
		
		# Load our training images from the data directory, performing data augmentation
		genTraining = keras.preprocessing.image.ImageDataGenerator(
			preprocessing_function=vgg16.preprocess_input,
			validation_split=VALIDATION_SPLIT,
			rotation_range = 360,
			horizontal_flip = True,
			vertical_flip = True
		)
		trainingData = genTraining.flow_from_directory(
			Configuration.path('classification_data'),
			subset = 'training',
			batch_size = BATCH_SIZE,
			target_size = VGG_IMAGE_SIZE
		)
		
		# Load our validation images from the data directory
		validationData = ClassificationNetwork.getValidationData()
		
		# Determine the number classes present in the training data
		numClasses = len(trainingData.class_indices)
		
		# Compute the weights for each of our outputs, based on the number of training samples present for each class
		# (CURRENTLY UNUSED, SINCE THIS WAS ACTUALLY PRODUCING LOWER PEAK VALIDATION ACCURACY VALUES)
		classWeights = TrainingUtils.computeWeights(trainingData)
		
		# Create the modified VGG16 model
		model = ClassificationNetwork.create(numClasses)
		
		# Compile the model with our desired optimisation algorithm and learning rate
		model.compile(
			loss = 'categorical_crossentropy',
			optimizer = keras.optimizers.RMSprop(),
			metrics = ['accuracy']
		)
		
		# Perform transfer learning and train the model, saving the best checkpoint to disk
		TrainingUtils.trainWithCheckpoints(
			model,
			'classification',
			trainingData.class_indices,
			trainingData,
			validationData,
			BATCH_SIZE,
			EPOCHS,
			'val_acc'
		)
