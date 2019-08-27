import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams["image.aspect"]="equal"
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt
from PIL import Image
import torch
from loss_functions import FocalLoss
import torch.nn as nn
from torch.autograd import Variable
from eye_dataset import *
import torch.nn as nn
import os
import glob


def normalization(image, maxi=255, mini=0, norm_max=None, norm_min=None):
    """Normalization to range of [0, 255]
    Args :
        image : numpy array of image
    Return :
        image : numpy array of image with values turned into standard scores
    """
    if norm_max is None:
        norm_max = np.max(image)
    if norm_min is None:
        norm_min = np.min(image)
    image_new = (image - norm_min)*(maxi - mini)/(norm_max-norm_min) + mini
    return image_new


def single_pass(model, image):
    """Run the model forward one time
        Args:
            model: loaded model (in this case CleanU_Net)
            image (Tensor): 1x572x572 image loaded
        Return:
            output (Tensor): 1x2x388x388 output
    """
    # expand the dimension to 1x1x572x572 to fit the model
    image = Variable(image).cuda()
    output = model(image)
    # print(output.size())
    return output


def softmax(output_image):
    """
        Args:
            output_image (Tensor): 1x2x388x388
        Return:
            output_image (numpy): 2x388x388
    """
    m = nn.Softmax2d()
    output_image = m(output_image).cpu().detach().numpy()
    return output_image


def mask_onehot(mask):
    """express mask as one_hot representation
    Args:
        mask (Tensor): HxW
    Return:
        mask (numpy): CxHxW
    """
    mask_1 = mask.numpy()  # white
    mask_0 = np.abs(mask_1 - 1)  # black
    mask = np.stack((mask_0, mask_1))
    return mask, mask_0, mask_1


def prediction(softmax_score, image_name=None, image_path=None):
    """generate predicted output_image
    Args:
        softmax_score(numpy): output from softmax function (cxHxW)
    Return:
        pred_image (numpy): predicted image (HxW)
    """
    pred_image = normalization(np.argmax(softmax_score, axis=0)).astype(np.uint8)
    return pred_image


def cross_entropy_loss(output, target):
    """generate the cross_entropy_loss in matrix
    Args:
        images (numpy): the generated output image (CxHxW)
        masks_onehot (numpy): the true mask (CxHxW)
    Return:
        CEL (numpy): CEL in matrix (388x388)

    """
    # class_weights = torch.tensor([1, 3]).float()  # np.asarray([1, 2])
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    CEL = criterion(output, target).cpu().detach().numpy()[0, :, :]
    return CEL


def confidence(softmax_score, mask_0, mask_1, difference=False):
    """generate the confidence in matrix
    Args:
        images (numpy): the generated output image (CxHxW)
        mask_0 (numpy): the true mask for background (HxW)
        mask_1 (numpy): the true mask for forground (HxW)
    Return:
        conf, conf_white, conf_black (numpy): confidence matrix (CxHxW)

    """
    conf_black = np.multiply(softmax_score[0, :, :], mask_0)
    conf_white = np.multiply(softmax_score[1, :, :], mask_1)
    conf = conf_black + conf_white
    if difference == True:
        return conf_black, conf_white
    else:
        return conf


def focal_loss(output, target, gamma=1):
    """generate the Mean_square_error in matrix
    Args:
        images (numpy): the generated output image (CxHxW)
        masks_onehot (numpy): the true mask (CxHxW)
    Return:
        MSE (numpy): MSE in matrix (388x388)

    """
    criterion = FocalLoss(gamma=gamma, reduce=False)
    FL = criterion(output, target).cpu().detach().numpy()[0, :, :]
    return FL


def difference_map(pred_im, true_mask, image_name=None, image_path=None):
    sum_im = np.zeros((3, 388, 388))
    for x in range(sum_im.shape[1]):
        for y in range(sum_im.shape[2]):
            if pred_im[x][y] > 0 and true_mask[x][y] > 0:
                sum_im[0][x][y] = 255
                sum_im[1][x][y] = 255
            elif true_mask[x][y] > 0:
                sum_im[0][x][y] = 255
            elif pred_im[x][y] > 0:
                sum_im[1][x][y] = 255
    sum_im = np.uint8(sum_im.transpose(1, 2, 0))
    return sum_im


def save_image(image, saved_as, save_path):
    image = Image.fromarray(image)
    # organize images in every epoch
    desired_path = save_path + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(saved_as)
    image.save(desired_path + export_name)


def apply_colormap_on_image(activation, colormap_name="seismic"):
    """(FROM Utku's Github Repository)
        Apply heatmap on image
    Args:
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    no_trans_heatmap = (no_trans_heatmap*255).astype(np.uint8)
    return no_trans_heatmap

    # import matplotlib.pyplot as plt


def apply_colormap_on_image_with_color_bar(activation, colormap_name="seismic", save_name=""):
    """
        Apply heatmap on image
    Args:
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    # Change alpha channel in colormap to make sure original image is displayed
    plt.ioff()
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # I don't know why but in pcolor, the image is flipped up-side-down thus I used np.flipud
    
    plt.imshow(activation,cmap=colormap_name)
    plt.colorbar()#orientation='horizontal', shrink=0.6)
    plt.clim(0,1)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', format='png')
    plt.close()


if __name__ == '__main__':
    # image loading
    inputs = EyeDatasetVal('../data/tr_images/', '../data/clean_masks/')

    # get one image from the folder
    _, image, mask = inputs.__getitem__(20)

    # add batch dimension to do foward pass
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0).cuda()

    # load the model from the path
    model = torch.load("../eye_pretrained_model.pt")

    # do the single pass
    output = single_pass(model, image)

    # generating softmax score
    softmax_score = softmax(output)[0, :, :, :]

    folder_name = "../output_images/original/"

    # one_hot represenetation of the mask
    _, mask_0, mask_1 = mask_onehot(mask.cpu()[0])

    pred_img = prediction(softmax_score)  # make prediction
    conf = confidence(softmax_score, mask_0, mask_1)  # make confidence map
    CE = cross_entropy_loss(output, mask)  # make Cross Entropy loss map
    FL_1 = focal_loss(output, mask, gamma=1)  # make Focal Loss map (gamma =1)
    FL_2 = focal_loss(output, mask, gamma=2)  # make Focal Loss map (gamma =2)
    FL_5 = focal_loss(output, mask, gamma=5)  # make Focal Loss map (gamma =5)
    # make difference of prediction and true mask as image
    diff = difference_map(pred_img, np.asarray(mask)[0])
    """
    # put all losses in single np array
    color_map = np.zeros([388*2, 388*2])
    color_map[0:388, 0:388] = CE
    color_map[388:, 0:388] = FL_1
    color_map[0:388, 388:] = FL_2
    color_map[388:, 388:] = FL_5
    color_map = normalization(color_map, 1, 0)
    color_map = apply_colormap_on_image(color_map)
    save_image(color_map, "color_map.png", folder_name)
    """
    norm_max = np.max([CE, FL_1, FL_2, FL_5])
    norm_min = np.min([CE, FL_1, FL_2, FL_5])
    print(norm_max, norm_min)
    # Normalization

    CE = normalization(CE, 1, 0, norm_max, norm_min)
    FL_1 = normalization(FL_1, 1, 0, norm_max, norm_min)
    FL_2 = normalization(FL_2, 1, 0, norm_max, norm_min)
    FL_5 = normalization(FL_5, 1, 0, norm_max, norm_min)

    apply_colormap_on_image_with_color_bar(1-conf, "seismic", folder_name+"conf.png")
    apply_colormap_on_image_with_color_bar(FL_1, "seismic", folder_name+"FL_1.png")
    """
    # apply color map
    conf = apply_colormap_on_image(1-conf, "seismic")
    CE = apply_colormap_on_image(CE, "jet")
    FL_1 = apply_colormap_on_image(FL_1, "jet")
    FL_2 = apply_colormap_on_image(FL_2, "jet")
    FL_5 = apply_colormap_on_image(FL_5, "jet")

    # save the outputs
    save_image(np.uint8(np.array(mask)[0]*255), "true_mask.png", folder_name)  # save true mask
    save_image(pred_img, "prediction.png", folder_name)
    save_image(diff, "difference.png", folder_name)
    save_image(conf, "confidence.png", folder_name)
    save_image(CE, "cross_entropy.png", folder_name)
    save_image(FL_1, "focal_loss_g1.png", folder_name)
    save_image(FL_2, "focal_loss_g2.png", folder_name)
    save_image(FL_5, "focal_loss_g5.png", folder_name)
    """
