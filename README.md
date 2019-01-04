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

### Updates

* 4/1/19: A bug was fixed and the results have improved. New features and labels files uploaded for URFD (links below). Optical flow components used for the training and the checkpoints to load the model trained on these images are also to the repository.

### Getting started

The repository contains the following files:

* **temporalnetxxx.py**: there are four scripts with this name, three of those are specific to a dataset (URFD, FDD and Multicam) and the remaining one, **temporalnetgeneral.py**, is the code for the experiment shown in the section 4.6 of the paper (a multi-tasking training with the previous three datasets). The four scripts first feed optical flow stacks through a VGG16 network to extract features (downloadable below). Then, they fine-tune two fully-connected layers with those features to correctly classify falls and daily living actions (not falls).

* **brightness.py** contains the script to generate the images used in the experiment of section 4.5 of the paper.

* **requirements.txt**: file with all the necessary python packages (install via pip).

___

### Reproducing the experiments

Necessary files if you want to re-train the network or use your own dataset:

* [Weights of the VGG16 network](https://drive.google.com/file/d/0B4i3D0pfGJjYNWxYTVUtNGtRcUE/view?usp=sharing). Weights of a pre-trained VGG16 in the UCF101 Action Recognition dataset using optical flow stacks. In the temporalneturfd.py, temporalnetfdd.py, temporalnetmulticam.py and temporalnetgeneral.py scripts, there is a variable called 'vgg_16_weights' to set the paths to this weights file.
* [Mean file](https://drive.google.com/file/d/1pPIArqld82TgTJuBHEibppkc-YLvQmVk/view?usp=sharing). In the temporalneturfd.py, temporalnetfdd.py, temporalnetmulticam.py and temporalnetgeneral.py scripts there is a variable called 'mean_file' to set the paths to this file. Used as a normalisation for the input of the network.

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

After that we include some variables related to the training of the network, such as the batch size or the weight to be applied to the class 0. Then, the variables 'save_plots' and 'save_features', to save the plots created during training and validation and to save the weights extracted from the VGG16, respectively. 'save_features' must be set to True or False depending on what you want to do:

1. If you use your own dataset or you want to extract the features by yourself, set 'save_features' to True the first time that the script is used and False after that. Take into account that if you run the script with the True value set, previous features will be erased.

2. If you want to load the extracted features, set 'save_features' to False.

If you have your model already trained, you can load the checkpoints and only do the evaluation part. Set the 'use_checkpoint' variable to True in that case. This skips the training part and obtains the results for each fold, computing at the end the average results.

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

### Optical flow images

We make downloadable the optical flow components used for the experiments if you want to re-train the network:

* [URFD optical flow images](https://drive.google.com/file/d/1D26r6xL7--GOByvE_fLspFFaOmp-s8Nl/view?usp=sharing)

### Checkpoints

In order to reproduce the results, we make the checkpoints of each fold available here:

* URFD [[Fold1](https://drive.google.com/file/d/1st02xocW_PZvadLHCPcOAIsXZgrU_t88/view?usp=sharing)][[Fold2](https://drive.google.com/file/d/1VGpJjR4nzE4jciegAuvXfz37rPLPG0I0/view?usp=sharing)][[Fold3](https://drive.google.com/file/d/1gqVJjx41FI0NxoYOU2swvruyYcP_Gakz/view?usp=sharing)][[Fold4](https://drive.google.com/file/d/1mESNjMKZmig7LzHTb85I99YKqN7vxz4q/view?usp=sharing)][[Fold5](https://drive.google.com/file/d/1d-aHtjXl7j57a9Rn3Q4iAbec5ElYM2mw/view?usp=sharing)][[All](https://drive.google.com/file/d/1P_vxqwc3lczgX5xwTLI_1K9yRii9OWKR/view?usp=sharing)]
