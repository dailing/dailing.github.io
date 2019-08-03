---
layout: post
title:  "Polygon-RNN and Other Interactive Object Segmentation Methods"
date:   2019-04-23 18:10:09 +0800
categories: note
---
The details of the method, my thoughts, and result are recorded in this post during my reproducing the two Polygon-RNN work.

# Type of interactive action used in interactive object selection

## Scribbles
Scribbles are used as seeds to model the appearance of foreground and background. Papers that used 
* Interactive graph cuts for optimal boundary & region segmentation of objects in ND images. In ICCV, 2001. 2
* An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. PAMI, 26(9):1124–1137, 2004. 2
* Video segmentation with just a few strokes. In ICCV, 2015. 2

## Bounding Boxes
GrubCut exploits annotations in the form of 2D bounding boxes, and then perform pixel level labeling with foreground/background models. Also 3D bounding boxes and point clouds are used to facilitate labeling.
* Beat the MTurkers: Automatic image labeling from weak 3D supervision. In CVPR, 2014.
* Active image segmentation propagation. In CVPR, 2016.
* Segmentation propagation in image-net. In ECCV, 2012.
* DeepCut: Object segmentation from bounding box annotations using convolutional neural networks. In arXiv:1605.07866, 2016
* GrabCut: Interactive foreground extraction using iterated graph cuts. In SIGGRAPH, 2004.

## Annotation at super-pixel level
Annotation at super-pixel make the annotation process more efficient.
* Semi-automatic video object segmentation by advanced manipulation of segmentation hierarchies. 2015.
* Parsing clothing in fashion photographs. In CVPR, 2012. 2

# RNN/Polygon based object segmentation
> Starting by detecting edge of a object, these methods find an optimal cycle that links the edge into a coherent region.

* Free-shape polygonal object localization. In ECCV, 2014
* Super-edge grouping for object localization by combining appearance and shape information. In CVPR, 2012.
* Towards large-scale city reconstruction from satellites. In ECCV, 2016


# Basic steps of using RNN to interactive segment

## Overall Polygon-RNN model

![Architecture](/asserts/post/2019-04-23-Polygon-RNN and Interactive Object Segmentation/Archetecture.png)

**The CNN architecture:**

A VGG+skip connection model is used to extract the visual features.
All features are used as input of RNN. RNN takes as input the previous state and the input image features to predict the next vertex. The vertex are represented and encoded using OneHot in a $$28 \times 28$$ binary map.

**RNN part**

Convolutional LSTM is used in the model.

This part is to be reported in detail after reading :
> Convolutional LSTM network: A machine
learning approach for precipitation now-casting. In NIPS,
pages 802–810, 2015.

**Predicting the first vertex:**

To predict the first vertex, another CNN with same arch is trained. The CNN takes as input the image and perform two tasks(Note that the second task are trained with the first one using an extra CONV layer.):
1. to predict the edge of the object (which can be calculated from segmentation map in the training set)
2. to predict the vertexes using original image and edge as input. The vertexes are also provided in the CityScape dataset.

# Reproduce Steps

1. Build the CNN part.
2. Train and test if edge detector and first vertex generator works.
2. Build the RNN part.
2. Overall Train and test.

# Results