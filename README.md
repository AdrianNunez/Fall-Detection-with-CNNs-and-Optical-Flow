## Fall-Detection-with-CNNs-and-Optical-Flow

This repository contains the code for our paper:

```
Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, I. (2017).
"Vision-Based Fall Detection with Convolutional Neural Networks"
Wireless Communications and Mobile Computing, 2017.
```

If you find the code useful for your research, please, cite our paper:

```
@article{nunez2017vision,
  title={Vision-Based Fall Detection with Convolutional Neural Networks},
  author={N{\'u}{\~n}ez-Marcos, Adri{\'a}n and Azkune, Gorka and Arganda-Carreras, Ignacio},
  journal={Wireless Communications and Mobile Computing},
  volume={2017},
  year={2017},
  publisher={Hindawi}
}

```

### Getting started

The repository contains the following files:

* **temporalnetxxx.py**: there are four scripts with this name, three of those are specific to a dataset (URFD, FDD and Multicam) and the remaining one, **temporalnetgeneral.py** it is the code for the experiment shown in the section 4.6 of the paper (a multi-tasking training with the previous three datasets). The four scripts do a feed-forward pass through the VGG16 of the optical flow stacks generated from the images of those datasets to get their features (downloadable below). Then, they fine-tune three fully-connected layers with those features to correctly classify actions.

* **brightness.py** contains the script to generate the images used in the experiment of section 4.5 of the paper.

* **requirements.txt**: file for installing all the necessary python packages.

___

### Reproducing the experiments

Necessary files:

* [Weights of the VGG16 network](https://drive.google.com/file/d/0B4i3D0pfGJjYNWxYTVUtNGtRcUE/view?usp=sharing). Weights of a pre-trained VGG16 in the UCF101 Action Recognition dataset using optical flow stacks. In the temporalneturfd.py, temporalnetfdd.py, temporalnetmulticam.py and temporalnetgeneral.py scripts there is a variable called 'vgg_16_weights' to set the paths to the weights, wherever you store them.
* [Mean file](https://drive.google.com/file/d/0B4i3D0pfGJjYTllxc0d2NGUyc28/view?usp=sharing). In the temporalneturfd.py, temporalnetfdd.py, temporalnetmulticam.py and temporalnetgeneral.py scripts there is a variable called 'mean_file' to set the paths to the weights, wherever you store them.

All the experiments were done under Ubuntu 16.04 Operating System, using Python 2.7 and OpenCV 3.1.0. All the necessary python packages have been included in the **requirements.txt** file. Use the following command to install them in your virtual environment:

```
pip install -r requirements.txt
```

#### 0. Using your own dataset

If you want to use your own dataset you need to have a directory with two subfolders: one called 'Falls' and the other one 'NotFalls'. Inside each of them there should be a folder for each fall/"not fall" video, where the images of the video are stored. Example of the directory tree used:

```
fall-dataset\
		Falls\
			fall0\
				flow_x_00001.jpg
				flow_x_00002.jpg
				...
		NotFalls\
			notfall0\
				flow_x_00001.jpg
				flow_x_00002.jpg
				...
```

#### 1. Download the code and change the paths

After downloading the code you will have to change the paths to adapt to your system and dataset. For each 'temporalnetxxx.py' script, in the very first lines, we have included several variables. Among them you can find 'data_folder', 'mean_file' and 'vgg_16_weights', that must point to the folder of your dataset, the [mean file](https://drive.google.com/file/d/0B4i3D0pfGJjYTllxc0d2NGUyc28/view?usp=sharing) of the network and the path to the [weights of the VGG16 network](https://drive.google.com/file/d/0B4i3D0pfGJjYNWxYTVUtNGtRcUE/view?usp=sharing). Then, to save the model and the weights after the training you have the paths 'model_file' and 'weights_file'. Finally, 'features_file' and 'labels_file' are the paths to the hdf5 files where the extracted features from the VGG16 CNN are going to be stored.

After that we include some variables related to the training of the network, such as the batch size or the weight to be applied to the class 0. Then, the variables 'save_plots' and 'save_features', to save the plots created during training and validation and to save the weights extracted from the VGG16, respectively. 'save_features' should be True the first time that the script is used and False after that. Take into account that if you run the script with the True value set, previous features will be erased. 

#### 2. Executing the code

Execute the code by calling:

```
python temporalneturfd.py
```

#### A. Reproducing the experiment with different lighting conditions (Section 4.5 of the paper)

'brightness.py' is required to darken or adding a lighting change to the original images of any dataset. 

1. Change the variables 'data_folder' and 'output_path' at the top of the script to match with the path to your dataset and the outputh path you desire (no need to be already created). By applying the script, the images inside 'data_folder' path will be transformed (darkened or dynamic light added to them) and will be stored in 'output_path'.

2. The 'mode' variable can have the value 'darken' to obtain the images for the experiment 4.5.1 (darkened images). Any other value will retrieve the images for the experiment 4.5.2 (lighting change).

3. You can also change the amount of darkness is added and the intensity of the added dynamic light applied. The variables 'darkness' and 'brightness' will control them, respectively.

### Extracted features and labels

These are the features extracted from the fc6 layer of the VGG16 network used for fine-tuning (vectors of size 4096). The instances are ordered so that the i-th element in the feature file has the i-th label of the labels file. For the Multicam dataset, the division inside each file is done by scenes, then by cameras.

* UR Fall Dataset (URFD)
  * [Features](https://drive.google.com/file/d/0B4i3D0pfGJjYa2dwclduMklLN2s/view?usp=sharing)
  * [Labels](https://drive.google.com/file/d/0B4i3D0pfGJjYcUhIM3pzQkV4dHM/view?usp=sharing)
* Multiple Cameras Fall Dataset (Multicam)*
  * [Features](https://drive.google.com/file/d/1Kfbm1RiKUr5q6S7Mq4LqTYGRyKyY_F91/view?usp=sharing) 
  * [Labels](https://drive.google.com/file/d/1krNC_QbGD4vE6XwEnuUdajtYy4_o4iaJ/view?usp=sharing)
* Fall Detection Dataset (FDD)
  * [Features](https://drive.google.com/file/d/0B4i3D0pfGJjYSXN6aW82MjhtSkE/view?usp=sharing)
  * [Labels](https://drive.google.com/file/d/0B4i3D0pfGJjYdTE4R2tYdHhLOXc/view?usp=sharing)

*Note: due to problems with data storage, we could not submit the original features used in the paper.
