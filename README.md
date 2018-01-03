# Fall-Detection-with-CNNs-and-Optical-Flow

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

We provide the following material to replicate the experiments.

## Necessary files

* [Weights of the VGG16 network](https://drive.google.com/file/d/0B4i3D0pfGJjYNWxYTVUtNGtRcUE/view?usp=sharing)
* [Mean file](https://drive.google.com/file/d/0B4i3D0pfGJjYTllxc0d2NGUyc28/view?usp=sharing)

## Extracted features and labels

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
