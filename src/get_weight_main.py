import torch
import torch.nn as nn
torch.manual_seed(99)
import numpy as np
from unet_model import SimpleUnet
from eye_dataset import EyeDataset, EyeDatasetTest
from torch.autograd import Variable
import copy
from misc_functions import *
from helper_modules import *


def get_model_weights(model):
    weight_list = []
    for module_pos, module in model._modules.items():
        # Only for upconvolution
        if ('up_' in module_pos and 'conv' not in module_pos) or ('conv_final' in module_pos):
            weight_list.append(copy.deepcopy(module.weight.detach().numpy()))
        # Sequential convs
        elif 'Sequential' in str(module):
            for single_layer in module:
                if 'Conv' in str(single_layer):
                    weight_list.append(copy.deepcopy(single_layer.weight.detach().numpy()))
    return weight_list


def find_weight_diff(wlist1, wlist2):
    diff_list = []
    print(wlist1[len(wlist1)-1][0][:5])
    print(wlist2[len(wlist2)-1][0][:5])
    for lweight1, lweight2 in zip(wlist1, wlist2):
        abs_diff_sum = np.sum(np.abs(np.float64(lweight1) - np.float64(lweight2)))
        diff_list.append(abs_diff_sum)
    return diff_list

    # create a new model with these weights
    model_rule = Net()
    model_rule.apply(weights_init_uniform_rule)


def initialize_weights(model, input_im):
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


def cnn_layer_visualization(model, input_im, selected_layer=2, selected_feature=1):

    optimizer = torch.optim.Adam([input_im], lr=0.1, weight_decay=1e-6)

    for i in range(20):
        print(i)
        x = input_im.cuda()
        n = 0
        for module_pos, module in model._modules.items():
            print(n)
            # Only for upconvolution
            if ('up_' in module_pos and 'conv' not in module_pos) or ('conv_final' in module_pos):
                # print("up_before: ", module.weight.detach().numpy()[0][0][0][0])
                x = module(x)
                print(module_pos, x.size())
                n += 1
                # print("n: ", n)
                # print("up_after: ", module.weight.detach().numpy()[0][0][0][0])
            elif "max" in module_pos:
                x = module(x)
                print(module_pos, x.size())
            # Sequential convs
            elif 'Sequential' in str(module):
                for single_layer in module:
                    if 'Conv' in str(single_layer):
                        # print("Conv_before weight: ", single_layer.weight.detach().numpy()[0][0][0][0])
                        # print("Conv_before bias: ", single_layer.bias.detach().numpy()[0])
                        x = single_layer(x)
                        print(str(single_layer), x.size())
                    elif "ReLU" in str(single_layer):
                        print(str(single_layer), x.size())
                        x = single_layer(x)
                        n += 1
                    if n == selected_layer:
                        break
            if n == selected_layer:
                break
        output = x[0, selected_feature]
        loss = -torch.mean(output)
        loss.backward()
        # Update image
        optimizer.step()
        new_im = recreate_image(input_im)
        # Save image
        if i % 5 == 0:
            im_path = '../generated/layer_vis_l' + str(selected_layer) + \
                '_f' + str(selected_feature) + '_iter' + str(i) + '.jpg'
            save_image(new_im, im_path)


if __name__ == '__main__':
    train_dataset = EyeDatasetTest('../data/tr_images',
                                   '../data/clean_masks')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=1,
                                               batch_size=1,
                                               shuffle=True)
    """
    model2 = SimpleUnet()
    initialize_weights(model2)
    prev_w = get_model_weights(model2)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizerd
    optimizer = torch.optim.SGD(model2.parameters(), lr=1)
    """
    model = load_model('../eye_pretrained_model.pt')
    _, sample, label = train_dataset[1]
    sample.unsqueeze_(0)
    print(sample.size())

    cnn_layer_visualization(model, Variable(sample, requires_grad=True))

    """
    for i in range(50):
        print(model2.conv_final.bias.data)
        # forward
        out = model2(sample)
        # loss
        loss = criterion(out, label)
        # print(model2.conv_final.weight[0][:5])
        # backward
        loss.backward()
        optimizer.step()
        # print(model2.conv_final.weight[0][:5])

        after_w = get_model_weights(model2)
        diff_list2 = find_weight_diff(prev_w, after_w)
        for item in diff_list2:
            print(item)
    """
