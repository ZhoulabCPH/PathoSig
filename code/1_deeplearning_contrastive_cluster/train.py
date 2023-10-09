import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from dataset import *
import warnings
warnings.filterwarnings('ignore')
from models import resnet,model
import math
import torch.cuda.amp as amp
import torchvision
import argparse
from utils import yaml_config_hook, save_model

def get_learning_rate(optimizer):
    """
    Returns the learning rate of the optimizer.
    """
    return optimizer.param_groups[0]['lr']

############################################################
####### Validation
############################################################

def initialization():
    """
    Initializes and returns the empty arrays for storing results.
    """
    Y_Sample_Name_List = np.array(0, dtype=np.str)
    Y_Weight_List = np.array([0]).reshape(-1, 1)
    Y_Pre_List = np.array([0]).reshape(-1, 1)
    Y_True_List = np.array([0]).reshape(-1, 1)
    return Y_Sample_Name_List, Y_Weight_List, Y_Pre_List, Y_True_List

def Get_Evaluation_Metrics(output, batch):
    """
    Extracts evaluation metrics from the output and batch data.
    Returns the weight, predicted values, and sample names.
    """
    Weight = np.array(output['Cluste'].detach().cpu())
    y_Pre = np.array(output['probability'].detach().cpu())
    y_name = np.array(batch['name'])
    return Weight, y_Pre, y_name

def validate(net, valid_loader, epoch):
    """
    Performs validation on the network using the validation data loader.
    Returns the validation loss.
    """
    valid_num = 0
    valid_loss = 0
    loss_instance_ = 0
    loss_cluster_ = 0
    torch.save({
        'state_dict': net.state_dict(),
        'epoch': epoch,
    }, out_dir + '/%08d.models.pth' % (epoch))
    net = net.eval()
    for t, batch in enumerate(valid_loader):
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()
            batch['image_Argument'] = batch['image_Argument'].cuda()
            output = net(batch)
            loss_instance = output['loss_instance'].mean()
            loss_cluster = output['loss_cluster'].mean()
            loss_s = loss_instance + loss_cluster
            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()
        valid_num += batch_size
        valid_loss += batch_size * loss_s.item()
    sum_train_loss = valid_loss / valid_num
    loss_instance_su = loss_instance_ / valid_num
    loss_cluster__su = loss_cluster_ / valid_num
    print('[------Step:{}Train_Sum:{}--Train_instance:{}--Train_cluster:{}--------]'.format(str(epoch),
                                                                                             str(round(sum_train_loss, 6)),
                                                                                             str(round(loss_instance_su, 6)),
                                                                                             str(round(loss_cluster__su, 6))))
    return sum_train_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    out_dir = args.out_dir
    initial_checkpoint = args.initial_checkpoint
    start_lr = float(args.start_lr)
    batch_size = int(args.batch_size)
    Epoch = int(args.Epoch)

    train_df, test_df = make_fold_WSI(args)

    train_dataset = HubmapDataset(train_df, args, train_augment5b, train_augment5a)
    test_dataset = HubmapDataset(test_df, args, valid_augment5, train_augment5b)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=int(args.workers),
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=2,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=null_collate,
    )
    res = resnet.get_resnet("ResNet50")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net = model.Net(arg=args, resnet=res).to(device)
    if initial_checkpoint != 'None':
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=False)  # True
    else:
        start_iteration = 0
        start_epoch = 0
        net.load_pretrain()  # pretrain
    num_iteration = Epoch * len(train_loader)

    lr_schedule = lambda t: np.interp([t], [0, num_iteration * 1 // 10, num_iteration],
                                      [start_lr, start_lr/10, start_lr/10*math.cos(((t+1-(num_iteration * 1 // 10))/(num_iteration-(num_iteration * 1 // 10)))*math.pi/2)])[0]

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, weight_decay=args.weight_decay)
    sum_train = 0
    sum_train_loss = 0
    sum_instance_loss = 0
    sum_cluster_loss = 0
    Train_nums = 0
    Valid_nums = 0
    Train_Name_List, Train_Weight_List, Train_Pre_List, Train_True_List = initialization()
    iteration = 0
    for i in range(Epoch):
        for t, batch in enumerate(train_loader):
            if (t % (len(train_loader)-1) == 0 and t != 0):
                sum_train_loss = validate(net, test_loader, i)
                Valid_nums += 1
                pass
            rate = get_learning_rate(optimizer)
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()
            batch['image_Argument'] = batch['image_Argument'].cuda()
            net.train()
            net.output_type = ['loss', 'inference']

            output = net(batch)
            loss_instance = output['loss_instance'].mean()
            loss_cluster = output['loss_cluster'].mean()
            loss_barlow = output['loss_barlow'].mean()
            loss = (loss_instance + loss_cluster)*(1-args.alpha)+loss_barlow*args.alpha

            optimizer.zero_grad()
            lr = lr_schedule(iteration)
            optimizer.param_groups[0].update(lr=lr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            loss_barlow_loss = loss_barlow.item()*args.alpha
            batch_loss = loss.item()
            instance_loss = loss_instance.item()
            cluster_loss = loss_cluster.item()
            sum_train_loss += batch_loss
            sum_instance_loss += instance_loss
            sum_cluster_loss += cluster_loss
            sum_train += 1
            if t % 10 == 0:
                sum_train_loss = sum_train_loss / sum_train
                sum_instance_loss_ = sum_instance_loss / sum_train
                sum_cluster_loss_ = sum_cluster_loss / sum_train
                print('Epoch:[{}/{}]\tStep:[{}/{}]\tlr:{}\tinstance_Loss:{}\tcluster_Loss:{}\tloss_barlow:{}'.format(i, Epoch, t, len(train_loader), str(round(lr, 6)), str(round(sum_instance_loss_, 6)),
                                                                                                                     str(round(sum_cluster_loss_, 6)), str(round(loss_barlow_loss, 6))))
                sum_train_loss = 0
                loss_barlow_loss = 0
                sum_instance_loss = 0
                sum_cluster_loss = 0
                sum_train = 0
                Train_nums += 1
            iteration += 1
            start_iteration = iteration
        start_epoch = i
        torch.cuda.empty_cache()
