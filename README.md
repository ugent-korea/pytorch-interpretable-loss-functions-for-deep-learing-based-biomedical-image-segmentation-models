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

## Experimental setups <a name="experimental_setup"></a>



## Results <a name="results"></a>

<table style="width:99%">
	<tr> 
		<th>Optimizer</th>
	    	<th>Learning Rate</th>
	    	<th>Lowest Loss</th>
	    	<th>Epoch</th>
		<th>Highest Accuracy</th>
	    	<th>Epoch</th>
	</tr>
	<tr>
		<th rowspan="3">SGD</th>
		<td align="center">0.001</td>
		<td align="center">0.196972</td>
		<td align="center">1445</td>
		<td align="center">0.921032</td>
		<td align="center">1855</td>
	</tr>
	<tr>
		<td align="center">0.005</td>
		<td align="center">0.205802</td>
		<td align="center">1815</td>
		<td align="center">0.918425</td>
		<td align="center">1795</td>
	</tr>
	<tr>
		<td align="center">0.01</td>
		<td align="center">0.193328</td>
		<td align="center">450</td>
		<td align="center">0.922908</td>
		<td align="center">450</td>
	</tr>
	<tr>
		<th rowspan="3">RMS_prop</th>
		<td align="center">0.0001</td>
		<td align="center">0.203431</td>
		<td align="center">185</td>
		<td align="center">0.924543</td>
		<td align="center">230</td>
	</tr>
	<tr>
		<td align="center">0.0002</td>
		<td align="center">0.193456</td>
		<td align="center">270</td>
		<td align="center">0.926245</td>
		<td align="center">500</td>
	</tr>
	<tr>
		<td align="center">0.001</td>
		<td align="center">0.268246</td>
		<td align="center">1655</td>
		<td align="center">0.882229</td>
		<td align="center">1915</td>
	</tr>
	<tr>
		<th rowspan="3">Adam</th>
		<td align="center">0.0001</td>
		<td align="center">0.194180</td>
		<td align="center">140</td>
		<td align="center">0.924470</td>
		<td align="center">300</td>
	</tr>
	<tr>
		<td align="center">0.0005</td>
		<td align="center">0.185212</td>
		<td align="center">135</td>
		<td align="center">0.925519</td>
		<td align="center">135</td>
	</tr>
	<tr>
		<td align="center">0.001</td>
		<td align="center">0.222277</td>
		<td align="center">165</td>
		<td align="center">0.912364</td>
		<td align="center">180</td>
	</tr>
		
</table>       


We chose the best learning rate that fits the optimizer based on **how fast the model converges to the lowest error**. In other word, the learning rate should make model to reach optimal solution in shortest epoch repeated. However, the intersting fact was that the epochs of lowest loss and highest accuracy were not corresponding. This might be due to the nature of loss function (Loss function is log scale, thus an extreme deviation might occur). For example, if the softmax probability of one pixel is 0.001, then the -log(0.001) would be 1000 which is a huge value that contributes to loss.
For consistency, we chose to focus on accuracy as our criterion of correctness of model. 



<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Accuracy of prediction</td>
	    </tr>
		<tr>
			<td width="99%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/graph_accuracy.png"> </td> 
		</tr>
	</tbody>
</table>       
We used two different optimizers (SGD, RMS PROP, and Adam). In case of SGD the momentum is manually set (0.99) whereas in case of other optimizers (RMS Prop and Adam) it is calculated automatically. 

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Accuracy of prediction</td>
	    </tr>
		<tr>
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/SGD_graph.png"> </td> 
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/RMS_graph.png"> </td>
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/Adam_graph.png"> </td>
		</tr>
		<tr>
			<td align="center">SGD<br />(lr=0.01,momentum=0.99)</td>
			<td align="center">RMS prop<br />(lr=0.0002)</td>
			<td align="center">Adam<br />(lr=0.0005)</td>
      		</tr>
	</tbody>
</table>       

### Model Downloads

Model trained with SGD can be downloaded via **dropbox**:
https://www.dropbox.com/s/ge9654nhgv1namr/model_epoch_2290.pwf?dl=0


Model trained with RMS prop can be downloaded via **dropbox**:
https://www.dropbox.com/s/cdwltzhbs3tiiwb/model_epoch_440.pwf?dl=0


Model trained with Adam can be downloaded via **dropbox**:
https://www.dropbox.com/s/tpch6u41jrdgswk/model_epoch_100.pwf?dl=0




### Example

<p align="center">
  <img width="250" height="250" src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_img.png"> <br /> Input Image</td>
</p>

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="5"><strong>Results comparsion</td>
	    </tr>
		<tr>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_mask.png"> </td>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_RMS.png"> </td>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_SGD.png"></td> 
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_Adam.png"> </td>
		</tr>
		<tr>
			<td align="center">original image mask</td>
			<td align="center">RMS prop optimizer <br />(Accuracy 92.48 %)</td>
			<td align="center">SGD optimizer <br />(Accuracy 91.52 %)</td>
			<td align="center">Adam optimizer <br />(Accuracy 92.55 %)</td>
      		</tr>
	</tbody>
</table>       

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
