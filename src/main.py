import torch
import torch.nn as nn
from loss_functions import FocalLoss
from unet_model import SimpleUnet
from unet_model_generalized import CleanU_Net
from eye_dataset import EyeDataset, EyeDatasetTest, EyeDatasetVal
from helper_modules import *
import random
random.seed(99)
torch.manual_seed(99)


if __name__ == '__main__':

    train_dataset = EyeDataset('../data/tr_images', '../data/clean_masks')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=8,
                                               batch_size=4,
                                               shuffle=True)

    tr_test_dataset = EyeDatasetVal('../data/tr_images', '../data/clean_masks')
    tr_test_loader = torch.utils.data.DataLoader(dataset=tr_test_dataset,
                                                 num_workers=1,
                                                 batch_size=1,
                                                 shuffle=False)

    test_test_dataset = EyeDatasetVal('../data/ts_images', '../data/clean_masks')
    test_test_loader = torch.utils.data.DataLoader(dataset=test_test_dataset,
                                                   num_workers=1,
                                                   batch_size=1,
                                                   shuffle=False)

    # Batch size was 4 for Adam experiments
    gpu_id = 1
    folder = "CE_13_Adam_init_test"
    epoch = 50
    # CSV File
    # model save
    suffix = '/train'
    model_dir = "../result/"+folder+"/saved_model/"
    prediction_dir = "../result/"+folder+suffix+"/prediction"
    csv_dir = "../result/"+folder+suffix+"/csv"

    model = SimpleUnet()
    #model = CleanU_Net(3, 2)
    #initialize_weights(model)
    model.cuda(gpu_id)

    #model = load_model('../eye_pretrained_model.pt')

    # Loss function
    class_weights = torch.tensor([1, 3]).float().cuda(gpu_id)  # np.asarray([1, 2])
    criterion = nn.CrossEntropyLoss(class_weights).cuda(gpu_id)
    #criterion = FocalLoss(gamma=5).cuda(gpu_id)

    # Optimizerd
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # val_res = validate_model(model, train_loader, criterion, 'train_full_run', True)
    print("total epoch: ", epoch)
    print("you are training with ", folder, " function!")
    for i in range(epoch):
        print('epoch:', i)
        # Train 1 epoch
        train_model(model, train_loader, criterion, optimizer, csv_dir, gpu_id)
        #tr_res = validate_model(model, train_loader, criterion, i, False, gpu_id)
        #print('Train acc: ', tr_res[0], "Train Loss: ", tr_res[1])
        # Validation on test data
        if i % 1 == 0:
            # test run on train
            train_res = validate_model(model, tr_test_loader, criterion,
                                       i, True, prediction_dir, csv_dir, gpu_id)
            print('Train Loss: ', train_res)

            # update folder names to test
            suffix = '/test'
            prediction_dir = "../result/"+folder+suffix+"/prediction"
            csv_dir = "../result/"+folder+suffix+"/csv"
            # test run on test
            train_res = validate_model(model, test_test_loader, criterion,
                                       i, True, prediction_dir, csv_dir, gpu_id)
            print('Test Loss: ', train_res)

            # update folder names to train
            suffix = '/train'
            prediction_dir = "../result/"+folder+suffix+"/prediction"
            csv_dir = "../result/"+folder+suffix+"/csv"
            # save model
            if i % 1 == 0:
                save_models(model, model_dir, i)
        # Validation on train data
        # Validation on test data
