# Interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-models

**Members** : <a href="https://github.com/PyeongKim">PyeongEun Kim</a>

**Supervisors** : <a href="https://github.com/utkuozbulak">Utku Ozbulak</a>, Prof. Arnout Van Messem, Prof. Wesley De Neve

## Technical Report
<p align="justify">
Detailed information regarding this repository can be found in the technical report. The technical report can be downloaded from the dropbox. Click <a href="https://www.dropbox.com/s/1rxhgilcia8wi2y/technical%20report%20for%20Interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-models.pdf?dl=0">here</a> to download the file.
</p>


## Description
<p align="justify">
After the initial break-through performances of encoder-decoder style deep learning models, multiple loss
functions were proposed to improve the functionality and the effectiveness of these models in the
segmentation of biomedical images, such as cross-entropy loss, the focal loss, and the Dice loss. However,
despite their critical role, the researches on the interpretability of the loss functions are lacking. As a result,
no clear answer on which loss function is most suitable in training the models for biomedical image
segmentation. Thus, to enhance the understanding of the loss functions, we aim to investigate the nature of
different loss functions. Also, we will propose a visual tool to illustrate the loss
surfaces of different loss functions throughout the training of deep segmentation model.
</p>





## Table of Content

* [Dataset](#dataset)

* [Model](#model)

* [Loss functions](#lossfunctions)

* [Experimental setups](#experimental_setup)

* [Results](#results)

* [Visualization of loss surfaces](#visualization)

* [Dependency](#dependency)

* [References](#references)



## Dataset <a name="dataset"></a>

<p align="justify">
We obtained the dataset from the paper
<a href="https://www.researchgate.net/publication/272191210_Estimation_of_the_Relative_Amount_of_Hemoglobin_in_the_Cup_and_Neuroretinal_Rim_Using_Stereoscopic_Color_Fundus_Images">“Estimation of the Relative Amount of Hemoglobin in the Cup and Neuroretinal Rim Using Stereoscopic Color
Fundus Images”.</a> The example of the eye data and its corresponding ground truth mask is shown in
the figure below. The eye image is RGB, with the dimension of 428x569 (width x height). In the dataset, there are
159 eye images and corresponding ground truth masks for optic disks and other parts. White pixels in the
ground truth mask represent the optic disks area in the eye image and black pixels in the ground truth mask
represent the non-optic disk area of the eye image. The dataset is divided into 150 images in the training set
and 9 images in the test set for training and testing, respectively. Note that our data has a class imbalance
problem that pixels of optic disks are only 10% of the total image.</p>

<p align="center">
<img src="https://github.com/ugent-korea/Interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/data_segmentation.png" height="300"></p> 


## Model <a name="model"></a>

#### Architecture

The model that we used is U-net (<a href="https://arxiv.org/pdf/1505.04597.pdf">original paper</a>), one of the first groundbreaking encoder-decoder style deep learning-based
model for image segmentation. The architecture of U-net model is illustrated in the figure below.

<p align="center">
<img src="https://github.com/ugent-korea/Interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/U-net_model.png" height="400"></p> 

## Loss functions <a name="lossfunctions"></a>

In this project, we used the two most popular loss functions: **the cross-entropy loss** and **the focal loss** functions.

### Cross-entropy loss function

<img src="https://latex.codecogs.com/svg.latex?CEL&space;=&space;-log(p_t)" title="CEL = -log(p_t)" />
### Focal loss function

## Experimental setups <a name="experimental_setup"></a>



## Results <a name="results"></a>

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Accuracy of prediction</td>
	    </tr>
		<tr>
			<td width="99%" align="center"> <img src="https://github.com/ugent-korea/pytorch-interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/graph_accuracy.png"> </td> 
		</tr>
	</tbody>
</table>     

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Intersection-over-union (IOU)</td>
	    </tr>
		<tr>
			<td width="99%" align="center"> <img src="https://github.com/ugent-korea/pytorch-interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/graph_IOU.png"> </td> 
		</tr>
	</tbody>
</table> 

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Prediction confidence of black pixels</td>
	    </tr>
		<tr>
			<td width="99%" align="center"> <img src="https://github.com/ugent-korea/pytorch-interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/graph_conf_black.png"> </td> 
		</tr>
	</tbody>
</table> 

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Prediction confidence of black pixels</td>
	    </tr>
		<tr>
			<td width="99%" align="center"> <img src="https://github.com/ugent-korea/pytorch-interpretable-loss-functions-for-deep-learing-based-biomedical-image-segmentation-model/blob/master/readme_images/graph_conf_white.png"> </td> 
		</tr>
	</tbody>
</table> 

## Visualization of loss surfaces <a name="visualization"></a>



## Dependency <a name="dependency"></a>

Following modules are used in the project:

    * python >= 3.6
    * numpy >= 1.14.5
    * torch >= 0.4.0
    * PIL >= 5.2.0
    * scipy >= 1.1.0
    * matplotlib >= 2.2.2
   

## References <a name="references"></a> :

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf

[2] P.Y. Simard, D. Steinkraus, J.C. Platt. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, http://cognitivemedium.com/assets/rmnist/Simard.pdf
