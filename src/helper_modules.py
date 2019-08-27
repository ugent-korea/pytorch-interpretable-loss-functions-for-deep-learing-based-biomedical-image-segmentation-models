import numpy as np
from PIL import Image
import glob as gl
import csv
import os
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from loss_visualization import softmax, confidence, mask_onehot


def train_model(model, data_train, criterion, optimizer, csv_folder, gpu_id=0):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train()
    model.cuda(gpu_id)
    for batch, (_, images, masks) in enumerate(data_train):
        w_prev = get_model_weights(model)
        # if batch%10 == 0:
        #print('Batch:', batch, 'of', len(data_train))
        images = Variable(images.cuda(gpu_id))
        masks = Variable(masks.cuda(gpu_id))
        outputs = model(images)
        #print(masks.shape, outputs.shape)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
        w_after = get_model_weights(model)
        diff = find_weight_diff(w_after, w_prev)
        export_history(diff, csv_folder, "weight_difference.csv")


def validate_model(model, data_val, criterion, epoch, make_prediction=True, prediction_folder='../result/prediction', csv_folder="../result/csv", gpu_id=0):
    """
        Validation run
    """
    total_val_loss = 0
    total_val_acc = 0
    abw, ab, aw, cb, cw, IOU = [], [], [], [], [], []
    for batch, (image_name, image, mask) in enumerate(data_val):
        # if batch%10 == 0:
            #print('Batch:', batch, 'of', len(data_val))
        with torch.no_grad():
            # Put in variable
            image = Variable(image.cuda(gpu_id))
            mask = Variable(mask.cuda(gpu_id))
            # Forward
            output = model(image)
            # Calculate loss
            total_val_loss = total_val_loss + criterion(output, mask).cpu().item()
            # Prediction
            pred = torch.argmax(output, dim=1).float()
            softmax_score = softmax(output)
            # Make prediction
            batch_amt = pred.size()[0]
            pred = pred.cpu().numpy()

            for pred_id in range(batch_amt):
                pred_mask = save_prediction_image(pred[pred_id], str(batch)+'_'+image_name[pred_id],
                                                  epoch, prediction_folder, save_im=make_prediction)
                conf_black, conf_white = confidence_calculation(softmax_score[pred_id], mask)
                int_ov_un = intersection_over_union(mask, pred[pred_id])

            acc, acc_black, acc_white = accuracy_check(mask, pred[pred_id])
            abw.append(acc)
            ab.append(acc_black)
            aw.append(acc_white)
            cb.append(conf_black)
            cw.append(conf_white)
            IOU.append(int_ov_un)
    export_history(abw, csv_folder, "accuracy.csv")
    export_history(ab, csv_folder, "accuracy_black.csv")
    export_history(aw, csv_folder, "accuracy_white.csv")
    export_history(cb, csv_folder, "cofindence_black.csv")
    export_history(cw, csv_folder, "confidence_white.csv")
    export_history(IOU, csv_folder, "IOU.csv")
    return total_val_acc/(batch + 1), total_val_loss/(batch + 1)


def get_model_weights(model):
    weight_list = []
    for module_pos, module in model._modules.items():
        # Only for upconvolution
        if ('up_' in module_pos and 'conv' not in module_pos) or ('conv_final' in module_pos):
            weight_list.append(copy.deepcopy(module.weight.data.cpu().detach().numpy()))
        # Sequential convs
        elif 'Sequential' in str(module):
            for single_layer in module:
                if 'Conv' in str(single_layer):
                    weight_list.append(copy.deepcopy(
                        single_layer.weight.data.cpu().detach().numpy()))
    return weight_list


def find_weight_diff(wlist1, wlist2):
    diff_list = []
    # print(wlist1[len(wlist1)-1][0][:5])
    # print(wlist2[len(wlist2)-1][0][:5])
    for lweight1, lweight2 in zip(wlist1, wlist2):
        abs_diff_sum = np.sum(np.abs(np.float64(lweight1) - np.float64(lweight2)))
        diff_list.append(abs_diff_sum)
    return diff_list


def initialize_weights(model):
    for module_pos, module in model._modules.items():
        # Only for upconvolution
        if ('up_' in module_pos and 'conv' not in module_pos) or ('conv_final' in module_pos):
            # print("up_before: ", module.weight.detach().numpy()[0][0][0][0])
            n = np.sqrt(2/torch.numel(module.weight.data))
            nn.init.normal_(module.weight.data, mean=0.0, std=n)
            module.bias.data = module.bias.data * 0.0
            # print("n: ", n)
            # print("up_after: ", module.weight.detach().numpy()[0][0][0][0])
        # Sequential convs
        elif 'Sequential' in str(module):
            for single_layer in module:
                if 'Conv' in str(single_layer):
                    # print("Conv_before weight: ", single_layer.weight.detach().numpy()[0][0][0][0])
                    # print("Conv_before bias: ", single_layer.bias.detach().numpy()[0])
                    n = np.sqrt(2/torch.numel(single_layer.weight.data))
                    nn.init.normal_(single_layer.weight.data, mean=0.0, std=n)
                    single_layer.bias.data = single_layer.bias.data * 0.0
                    # print("n: ", n)
                    # print("Conv_After weight: ", single_layer.weight.detach().numpy()[0][0][0][0])
                    # print("Conv_After bias: ", single_layer.bias.detach().numpy()[0])


def write_to_csv(file_name, arr):
    file_to_write = open(file_name, 'a')
    for item in arr:
        file_to_write.write(str(item)+',')
    file_to_write.write('\n')
    file_to_write.close()
    return 1


def confidence_calculation(softmax_score, mask):
    _, mask_0, mask_1 = mask_onehot(mask.cpu()[0])
    conf_black, conf_white = confidence(softmax_score, mask_0, mask_1, True)
    return (conf_black.sum())/mask_0.sum(), (conf_white.sum())/mask_1.sum()


def intersection_over_union(mask, pred_mask):
    mask = mask.cpu().numpy()[0]
    intersection = np.multiply(mask, pred_mask)
    union = (mask + pred_mask) - intersection
    return intersection.sum()/(union.sum())


def save_prediction_image(pred_img, im_name, epoch, save_folder_name="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """
    # Disc. pred image
    # print(pred_img.shape)
    if len(pred_img.shape) == 3:
        pred_img = pred_img[0]
    pred_as_arr = pred_img * 255
    pred_as_arr[pred_as_arr > 100] = 255
    pred_as_arr[pred_as_arr <= 100] = 0
    pred_img = Image.fromarray(pred_as_arr.astype('uint8'))
    # Set path
    save_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if not exist

    # Save
    if save_im:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if 'png' not in str(im_name):
            export_name = str(im_name) + '.png'
        else:
            export_name = str(im_name)

        pred_img.save(save_path + export_name)
    return pred_as_arr


def recreate_save_image(out_im, save_path='result_images', export_name='sample', im_type='cell'):
    if im_type == 'cell':
        out_im = out_im[0]
    out_im = out_im*255
    out_im[out_im > 255] = 255
    out_im[out_im < 0] = 0
    # print(np.std(out_im))
    # print(np.mean(out_im))
    if len(np.shape(out_im)) == 2:
        # np.expand_dims(out_im, axis=0)
        out_im = np.asarray([out_im, out_im, out_im])
    out_im = out_im.transpose(1, 2, 0)
    out_im = Image.fromarray(out_im.astype('uint8'))
    out_im.save(save_path+'/'+export_name+'.png')
    return out_im


def accuracy_check(mask, pred):
    mask_1 = mask.cpu().numpy()[0]
    mask_0 = 1 - mask_1
    pred_1 = pred
    pred_0 = 1 - pred_1
    # Check same pixels
    #print(mask.shape, pred.shape)
    acc = np.equal(pred, mask_1).sum()/len(pred.flatten())
    white_acc = pred_1.sum()/mask_1.sum()
    black_acc = pred_0.sum()/mask_0.sum()

    return acc, black_acc, white_acc


def export_history(value, folder, file_name):
    """ export data to csv format
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder (list): folder path
        file_name: file name with path
    """
    # if folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = folder+"/"+file_name
    file_existence = os.path.isfile(file_name)

    # if there is no file make file
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # if there is file overwrite
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # close file when it is done with writing
    file.close()


def save_models(model, path, epoch):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path+"/model_epoch_{0}.pt".format(epoch))


def load_model(path_to_model):
    # Load torch model
    model = torch.load(path_to_model)
    return model


if __name__ == '__main__':
    pass
