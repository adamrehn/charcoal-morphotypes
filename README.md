# Fossil charcoal particle identification and classification by two convolutional neural networks

This repository contains the implementation of the charcoal particle segmentation and classification system presented in the journal article:

- Emma Rehn, Adam Rehn, and Aidan Possemiers. Fossil charcoal particle identification and classification by two convolutional neural networks. **(In review.)**


## Contents

- [Requirements](#requirements)
- [Obtaining the container image](#obtaining-the-container-image)
- [Obtaining the input data and training checkpoints](#obtaining-the-input-data-and-training-checkpoints)
- [Performing inference on input images](#performing-inference-on-input-images)
- [(Optional) Building the container image from source](#optional-building-the-container-image-from-source)
- [(Optional) Training the neural networks from scratch](#optional-training-the-neural-networks-from-scratch)
- [Details of the individual script files](#details-of-the-individual-script-files)
    - [Data preparation scripts](#data-preparation-scripts)
    - [Training scripts](#training-scripts)
    - [Inference scripts](#inference-scripts)
    - [Report generation scripts](#report-generation-scripts)
- [License](#license)


## Requirements

The code in this repository uses a GPU-accelerated Docker container to provide a reproducible environment within which to perform training and inference. Running the container image requires the following hardware and software:

- An NVIDIA graphics card [with CUDA support](https://developer.nvidia.com/cuda-gpus) and at least 8GB of VRAM
- 64-bit version of one of Docker's [supported Linux distributions](https://docs.docker.com/install/#supported-platforms) (CentOS 7+, Debian 7.7+, Fedora 26+, Ubuntu 14.04+)
- [Docker Community Edition (CE)](https://docs.docker.com/install/) version 19.03 or newer
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) version 2.2.2 or newer
- NVIDIA binary driver version 384.81 or newer


## Obtaining the container image

You can download a prebuilt version of the Docker container image by running the following command:

```bash
docker pull "adamrehn/charcoal-morphotypes:latest"
```

If you would like to build the container image from source instead, see the [Building the container image from source](#optional-building-the-container-image-from-source) section.


## Obtaining the input data and training checkpoints

The training and validation data for both neural networks is available for download from <https://doi.org/10.25903/5d006c1494cf9>:

1. Ensure you have downloaded or cloned a local copy of this repository.
2. Download and extract the archive file `preprocessed.7z`. You will need software that supports the 7-zip archive format.
3. Copy all of the extracted image files to the `data/segmentation/preprocessed` subdirectory of your local copy of this repository.

Pretrained checkpoints for both neural networks are available for download from [the releases page of this repository](https://github.com/adamrehn/charcoal-morphotypes/releases):

1. Ensure you have downloaded or cloned a local copy of this repository.
2. Download and extract the archive file `checkpoint-classification.zip`.
3. Download and extract the archive file `checkpoint-segmentation.zip`.
4. Copy all of the extracted files to the `checkpoints` subdirectory of your local copy of this repository.

If you would like to train the neural networks from scratch instead, see the [Training the neural networks from scratch](#optional-training-the-neural-networks-from-scratch) section.


## Performing inference on input images

1. Ensure you have downloaded or cloned a local copy of this repository.

2. Ensure you have [downloaded the training checkpoints for both neural networks](#obtaining-the-input-data-and-training-checkpoints) and extracted them to the `checkpoints` subdirectory of your local copy of the repository.

3. Ensure the image file(s) you wish to perform inference on are stored inside the root directory of your local copy of the repository, since this is the only filesystem directory from the host system that will be visible inside the Docker container.

4. Start the Docker container by running the shell script [docker/run.sh](./docker/run.sh) from a terminal. This will provide you with an interactive shell that allows you to run commands inside the container.

5. Perform an inference pass by running the inference script for the appropriate neural network:
    
    - To perform inference using the segmentation neural network, run the command `python3 infer_segmentation.py INFILE OUTFILE`, where `INFILE` is the input image filename and `OUTFILE` is the filename that should be used to store the result image.
    
    - To perform inference using the classification neural network, run the command `python3 infer_classification.py INFILE`, where `INFILE` is the input image filename. The list of classification probablities will be printed to standard output, which by default is simply displayed in the terminal window.

6. Once you are done, run the `exit` command to close the interactive shell. The Docker container will be stopped automatically.


## (Optional) Building the container image from source

**Note: building the Docker container image from source is purely optional. To use a prebuilt version of the image, see the details in the [Obtaining the container image](#obtaining-the-container-image) section.**

To build the container image from source, run the shell script [docker/build.sh](./docker/build.sh) from a terminal. This will first build a Keras base image and then build the container image for the code in this repository. Building the images may take some time to complete depending on the speed of your internet connection.


## (Optional) Training the neural networks from scratch

**Note: training the neural networks from scratch is purely optional. To use pretrained checkpoints for both networks, see the details in the [Obtaining the input data and training checkpoints](#obtaining-the-input-data-and-training-checkpoints) section.**

1. Ensure you have downloaded or cloned a local copy of this repository.

2. Ensure you have [downloaded the training and validation data for both neural networks](#obtaining-the-input-data-and-training-checkpoints) and extracted it to the `data/segmentation/preprocessed` subdirectory of your local copy of the repository.

3. Start the Docker container by running the shell script [docker/run.sh](./docker/run.sh) from a terminal. This will provide you with an interactive shell that allows you to run commands inside the container.

4. Slice the preprocessed training data by running the command `python3 dataprep_slice.py`.

5. Window the sliced training data by running the command `python3 dataprep_window.py`.

6. Copy the sliced training data into the appropriate subdirectories for each morphotype classification by running the command `python3 dataprep_morphotypes.py`.

7. Train the segmentation neural network by running the command `python3 train_segmentation.py`.

8. Train the classification neural network by running the command `python3 train_classification.py`. Note that even when training the classification neural network "from scratch", transfer learning is still used to accelerate the process.

9. Generate the validation report for the segmentation neural network by running the command `python3 report_segmentation.py`.

10. Generate the validation report for the classification neural network by running the command `python3 report_classification.py`.

11. Once you are done, run the `exit` command to close the interactive shell. The Docker container will be stopped automatically.


## Details of the individual script files

### Data preparation scripts

- [**dataprep_preprocess.py**](./dataprep_preprocess.py) - this script performs preprocessing for the segmentation neural network training data in the `data/segmentation/raw` directory and saves the preprocessed output to the `data/segmentation/preprocessed` directory. This preprocessed data is then ready to be sliced using the script `dataprep_slice.py`. **(Note that for filesize reasons the preprocessed training data is what has been made available for download, rather than the raw Photoshop images, so you won't ever need to run this script.)**

- [**dataprep_slice.py**](./dataprep_slice.py) - this script extracts individual charcoal particle images from the preprocessed data in the `data/segmentation/preprocessed` directory and saves the individual images to the `data/segmentation/sliced` directory. These individual images are then ready to be classified by a human so that they can be used as training data for the classification neural network. The individual images are also ready to be windowed using the script `dataprep_window.py`.

- [**dataprep_window.py**](./dataprep_window.py) - this script applies a sliding window to the individual charcoal particle images from the sliced data in the `data/segmentation/sliced` directory and saves the windowed images to the `data/segmentation/windowed` directory. These windowed images are then ready for use in training the segmentation neural network without any further modifications.

- [**dataprep_batch.py**](./dataprep_batch.py) - this script moves the individual images in the `data/segmentation/sliced` directory into numbered subdirectories containing batches of approximately 100 images each. This can be handy for dividing a large number of images into more manageable batches for human classification. **(Note that the results of human classification are already provided by the morphotypes preparation script, so you won't ever need to run this script.)**

- [**dataprep_morphotypes.py**](./dataprep_morphotypes.py) - this script reproduces the human classification that was performed by hand using the [Taxonomist](https://github.com/adamrehn/taxonomist) tool and copies individual images in the `data/segmentation/sliced` directory into the appropriate subdirectories for each morphotype in the `data/classification` directory.

### Training scripts

- [**train_segmentation.py**](./train_segmentation.py) - this script trains the segmentation neural network using the training data in the `data/segmentation/preprocessed` directory and saves a checkpoint of the most accurate trained model to the `checkpoints` directory.

- [**train_classification.py**](./train_classification.py) - this script trains the classification neural network using the training data in the `data/classification` directory and saves a checkpoint of the most accurate trained model to the `checkpoints` directory.

### Inference scripts

- [**infer_segmentation.py**](./infer_segmentation.py) - this script loads the last saved checkpoint of the segmentation neural network from the `checkpoints` directory and performs segmentation of the user-specified input image(s).

- [**infer_classification.py**](./infer_classification.py) - this script loads the last saved checkpoint of the classification neural network from the `checkpoints` directory and performs classification of the user-specified input image(s).

### Report generation scripts

- [**report_segmentation.py**](./report_segmentation.py) - this script loads the last saved checkpoint of the segmentation neural network from the `checkpoints` directory and performs validation against the network's validation dataset, generating a HTML report in the `reports/segmentation` directory.

- [**report_classification.py**](./report_classification.py) - this script loads the last saved checkpoint of the classification neural network from the `checkpoints` directory and performs validation against the network's validation dataset, generating a HTML report in the `reports/classification` directory.


## License

The code in this repository is licensed under the MIT License. See the file [LICENSE](./LICENSE) for details.
