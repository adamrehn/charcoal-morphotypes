ARG KERAS_VERSION=2.2.2
FROM keras:${KERAS_VERSION}

# Install GDAL, ImageMagick and OpenCV, along with our required Python packages
RUN conda install gdal=2.2.4 && \
	conda install -c conda-forge imagemagick=7.0.8_11 && \
	conda clean -yt
RUN pip install humanfriendly==4.16.1 maskutils==0.0.2 mergetiff==0.0.19 natsort==5.4.1 opencv-python==3.4.3.18 slidingwindow==0.0.13

# Pre-download the weights for VGG16
# (Note that we use the Theano backend in CPU mode, since Tensorflow will crash without GPU support)
COPY --chown=keras:1000 weights.py /tmp/weights.py
RUN KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 /tmp/weights.py

# Install the Keras SegNet implementation we use
RUN pip install git+git://github.com/ImmortalEmperor/keras-segnet.git@v0.0.1
