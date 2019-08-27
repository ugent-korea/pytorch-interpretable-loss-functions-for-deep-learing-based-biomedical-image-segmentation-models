import torch
import torch.nn as nn
from loss_functions import FocalLoss
from unet_model import SimpleUnet
from unet_model_generalized import CleanU_Net
from eye_dataset import EyeDataset, EyeDatasetTest, EyeDatasetVal
from helper_modules import *
from loss_visualization import *
random.seed(99)
torch.manual_seed(99)


def multiple_forward_pass(model, loss_function, epoch, input_data, true_mask, folder_path):
    """
        Args:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for iteration in range(epoch):
        input_data = Variable(input_data.cuda())
        true_mask = Variable(true_mask.cuda())
        output = model(input_data)
        # print(masks.shape, outputs.shape)
        print(iteration)
        if iteration % 100 == 0:
            folder_name = folder_path+"iteration_"+str(iteration)

            #output = model(input_data)
            softmax_score = softmax(output)[0, :, :, :]
            _, mask_0, mask_1 = mask_onehot(true_mask.cpu()[0])

            pred_img = prediction(softmax_score)  # make prediction
            conf = confidence(softmax_score, mask_0, mask_1)  # make confidence map
            CE = cross_entropy_loss(output, mask)  # make Cross Entropy loss map
            focal_dic = {}
            for g in list([1, 2, 3, 5]):
                loss = focal_loss(output, mask, gamma=g)  # make Focal Loss map (gamma =1)
                focal_dic[g] = loss  # make Focal Loss map (gamma =5)
            # make difference of prediction and true mask as image
            diff = difference_map(pred_img, np.asarray(mask)[0])

            norm_max = np.max([CE, focal_dic[1], focal_dic[2], focal_dic[3], focal_dic[5]])
            norm_min = np.min([CE, focal_dic[1], focal_dic[2], focal_dic[3], focal_dic[5]])
            print(iteration, norm_max, norm_min)
            # Normalization
            if str(loss_function) == "CrossEntropyLoss()":
                loss = normalization(CE, 1, 0, norm_max, norm_min)
                graph_name = "/Cross_Entropy_Loss.png"
            elif str(loss_function) == "FocalLoss()":
                gamma = int(folder_path[-2])
                loss = normalization(focal_dic[gamma], 1, 0, norm_max, norm_min)
                graph_name = "/Focal_Loss_g"+str(gamma)+".png"

            # save input image
            save_image(np.uint8(np.array(
                (input_data[0, :, :, :]*255).detach().cpu()).transpose(1, 2, 0)), "input_img.png", folder_name)
            # save true mask
            save_image(np.uint8(np.array(mask)[0]*255),
                       "true_mask.png", folder_name)
            # save prediction
            save_image(pred_img, "prediction.png", folder_name)
            # save difference map
            save_image(diff, "difference.png", folder_name)
            # save heatmap with colorbar
            apply_colormap_on_image_with_color_bar(
                1-conf, "seismic", folder_name+"/confidence.png")
            #print((1-conf).shape)
            apply_colormap_on_image_with_color_bar(
                loss, "jet", folder_name+graph_name)
        loss = loss_function(output, true_mask)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()


if __name__ == '__main__':
    random.seed(99)
    torch.manual_seed(99)
    gpu_id = 0
    epoch = 1001
    folder_dir = "../interpretability_iterations/"
    model_dir = "../result/Adam/Focal_5_Adam_init/saved_model/model_epoch_100.pt"
    inputs = EyeDatasetVal('../data/train_images/', '../data/clean_masks/')
    print(len(inputs))
    for i in range(len(inputs)):
        # get one image from the folder
        image_name, image, mask = inputs.__getitem__(i)

        # add batch dimension to do foward pass
        image = image.unsqueeze(0).cuda(gpu_id)
        mask = mask.unsqueeze(0).cuda(gpu_id)
        _, mask_0, mask_1 = mask_onehot(mask.cpu())
        white = np.sum(mask_1)
        black = np.sum(mask_0)
        w_white = black/(white+black)
        w_black = white/(white+black)
        #
        class_weights = torch.tensor([0.3, 0.9]).float().cuda(0)
        loss_folder_pair = [(nn.CrossEntropyLoss(class_weights), folder_dir+image_name+"/CrossEntropyLoss/"),
                            (FocalLoss(gamma=1), folder_dir+image_name+"/Focal_loss_g1/"),
                            (FocalLoss(gamma=2), folder_dir+image_name+"/Focal_loss_g2/"),
                            (FocalLoss(gamma=3), folder_dir+image_name+"/Focal_loss_g3/"),
                            (FocalLoss(gamma=5), folder_dir+image_name+"/Focal_loss_g5/")]

    # nn.CrossEntropyLoss()  # FocalLoss(gamma=1)  # nn.CrossEntropyLoss()
        for loss_function, folder_path in loss_folder_pair:
            model = torch.load(
                model_dir).cuda(gpu_id)
            print(folder_path)
            multiple_forward_pass(model, loss_function, epoch=epoch, input_data=image,
                                  true_mask=mask, folder_path=folder_path)
