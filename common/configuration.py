from os import path

# The full paths to each of our key directories
_PATHS = {}
_PATHS['root']                           = path.dirname(path.dirname(__file__))
_PATHS['checkpoints']                    = path.join(_PATHS['root'], 'checkpoints')
_PATHS['logs']                           = path.join(_PATHS['root'], 'logs')
_PATHS['data']                           = path.join(_PATHS['root'], 'data')
_PATHS['reports']                        = path.join(_PATHS['root'], 'reports')
_PATHS['classification_data']            = path.join(_PATHS['data'], 'classification')
_PATHS['segmentation_data']              = path.join(_PATHS['data'], 'segmentation')
_PATHS['segmentation_data_raw']          = path.join(_PATHS['segmentation_data'], 'raw')
_PATHS['segmentation_data_preprocessed'] = path.join(_PATHS['segmentation_data'], 'preprocessed')
_PATHS['segmentation_data_sliced']       = path.join(_PATHS['segmentation_data'], 'sliced')
_PATHS['segmentation_data_windowed']     = path.join(_PATHS['segmentation_data'], 'windowed')
_PATHS['classification_reports']         = path.join(_PATHS['reports'], 'classification')
_PATHS['segmentation_reports']           = path.join(_PATHS['reports'], 'segmentation')

class Configuration:
	
	@staticmethod
	def path(key):
		'''
		Returns the full path for the specified directory identifier
		'''
		return _PATHS[key]
