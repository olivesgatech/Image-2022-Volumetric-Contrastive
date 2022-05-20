from __future__ import print_function

import argparse

import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms, datasets


from util import adjust_learning_rate
from util import set_optimizer
from networks.resnet_big import SupConResNet_Semantic
from networks.models_deeplab_head import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm
from dataset import SeismicDataset
from sklearn.metrics import jaccard_score
import numpy as np
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

torch.backends.cudnn.enabled = False
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device ID')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--super', type=int, default=1,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--img_dir', type=str, default='',
                        help='path to images')
    parser.add_argument('--target_dir', type=str, default='',
                        help='path to labels')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--test_split', type=int, default=1,
                        help='test_split')
    parser.add_argument('--n_cls', type=int, default=6,
                        help='number_classes')
    parser.add_argument('--parallel', type=int, default=10,
                        help='test_split')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18_seismic')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','Seismic'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif(opt.dataset == 'Seismic'):
        opt.n_cls = 6
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt):
    # construct data loader

    if (opt.dataset == 'Seismic'):
        mean = .501
        std = .109
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform_target = transforms.Compose([
        transforms.ToTensor(),

    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    val_transform_target = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'Seismic':
        csv_path_train = './csv_files/train_base.csv'
        if(opt.test_split == 1):
            csv_path_test = './csv_files/test_fold_1.csv'
        elif(opt.test_split == 2):
            csv_path_test = './csv_files/test_fold_2.csv'
        else:
            csv_path_test = './csv_files/test_fold_3.csv'
        train_dataset = SeismicDataset(csv_path_train,opt.img_dir,opt.target_dir, transform=train_transform, target_transform=train_transform_target)
        val_dataset = SeismicDataset(csv_path_test,opt.img_dir,opt.target_dir, transform=val_transform,target_transform=val_transform_target)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_model(opt,model):

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
                new_state_dict[k].requires_grad = False
            state_dict = new_state_dict
        cudnn.benchmark = True
        outputchannels = opt.n_cls
        classifier = DeepLabHead(512, outputchannels)
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            param.requires_grad=False
        for param in model.named_parameters():
            print(param.requires_grad)
    return model, classifier


def train_fn_supcon(loader, model, classifier, optimizer, loss_fn, scaler,opt):
    loop = tqdm(loader)

    for batch_idx, (data, targets,_) in enumerate(loop):

        data = data.to(device=opt.device)

        targets = targets.long().squeeze(1).to(device=opt.device)

        input_shape = data.shape[-2:]

        with torch.cuda.amp.autocast():
            predictions = model(data)
            predictions = predictions['out']

            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # update tqdm loop
        loop.set_postfix(loss=loss.item())




def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    criterion = nn.CrossEntropyLoss()
    model_sup = SupConResNet_Semantic(name=opt.model)
    model_sup.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)



    model, classifier = set_model(opt, model_sup)
    model = _SimpleSegmentationModel(model, classifier)
    model = model.to(opt.device)
    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    scaler = torch.cuda.amp.GradScaler()

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch

        train_fn_supcon(train_loader, model, classifier, optimizer, criterion, scaler, opt)


        # eval for one epoch
    miou, acc = check_accuracy(val_loader, model, device=opt.device)


    with open("results.txt", "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(str(acc) + '\n')
        file.write(str(miou) + '\n')



def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    miou_list = []
    with torch.no_grad():
        for x, y,_ in tqdm(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            out = model(x)['out']


            output = out.argmax(1)

            preds = output[0]

            gt = y[0][0]

            k = jaccard_score(preds.detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten(),
                              average='macro')

            miou_list.append(k)

            num_correct += (preds == gt).sum()
            num_pixels += torch.numel(preds)

    pixel_acc = num_correct / num_pixels * 100
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")

    print("MIOU: " + str(np.mean(miou_list)))
    model.train()
    return np.mean(miou_list), pixel_acc

if __name__ == '__main__':
    main()
