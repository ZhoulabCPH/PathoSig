is_amp = True
import warnings

warnings.filterwarnings('ignore')
from torch.nn.functional import normalize
from dataset import *
from augmentation import *
import contrastive_loss
from my_variable_swin_v1 import *
from upernet import *
import matplotlib.pyplot as plt


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def Visual(x_i, x_j, batch):
    # x_i=batch['image']
    # x_j=batch['image_Argument']
    x_i = np.array(x_i.cpu().detach().numpy())
    x_j = np.array(x_j.cpu().detach().numpy())
    plt.subplots(figsize=(50, 50))
    plt.subplot(4, 2, 1)
    plt.imshow(np.transpose(x_i[0], (1, 2, 0)))
    plt.subplot(4, 2, 2)
    plt.imshow(np.transpose(x_i[1], (1, 2, 0)))
    plt.subplot(4, 2, 3)
    plt.imshow(np.transpose(x_j[0], (1, 2, 0)))
    plt.subplot(4, 2, 4)
    plt.imshow(np.transpose(x_j[1], (1, 2, 0)))
    plt.show()


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.5, 0.5, 0.5]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.5, 0.5, 0.5]  # [0.5, 0.5, 0.5]

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


############################################################
####### Configuration
############################################################

class Net(nn.Module):

    def load_pretrain(self, ):


        print('loading %s ...' % self.arg.Resnet_PRE_Path)
        checkpoint = torch.load(self.arg.Resnet_PRE_Path, map_location=lambda storage, loc: storage)
        print(self.res.load_state_dict(checkpoint, strict=False))  # True

    def __init__(self, arg, resnet):
        super(Net, self).__init__()
        self.res = resnet
        self.arg = arg
        self.cluster_num = self.arg.cluster_dim
        self.feature_dim = self.arg.feature_dim
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()
        self.Downsampling = nn.AvgPool2d((7, 7), stride=None)
        self.Laten_Dim = self.arg.Laten_Dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.Laten_Dim, self.Laten_Dim),
            nn.ReLU(),
            nn.Linear(self.Laten_Dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.Laten_Dim, self.Laten_Dim),
            nn.ReLU(),
            nn.Linear(self.Laten_Dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm1d(self.arg.Laten_Dim, affine=False)

    def forward(self, batch):
        # x_i = self.rgb(batch['image'])
        # x_j = self.rgb(batch['image_Argument'])
        x_i = batch['image']
        x_j = batch['image_Argument']
        # Visual(x_i,x_j,batch)
        h_i = self.res(x_i)
        h_j = self.res(x_j)
        ##Contrastive
        # h_i = self.Downsampling(self.encoder(x_i)[-1]).squeeze()
        # h_j = self.Downsampling(self.encoder(x_j)[-1]).squeeze()

        c = self.bn(h_i).T @ self.bn(h_j)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.arg.Laten_Dim)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss_barlow = on_diag + self.arg.lambd * off_diag
        ##instance
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        ##cluster
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        device = c_i.device
        criterion_instance = contrastive_loss.InstanceLoss(len(batch['index']), self.arg.instance_temperature,
                                                           device).to(device)
        criterion_cluster = contrastive_loss.ClusterLoss(self.cluster_num, self.arg.cluster_temperature, device).to(
            device)
        output = {}
        if 'loss' in self.output_type:
            output['loss_instance'] = criterion_instance(z_i, z_j)
            output['loss_cluster'] = criterion_cluster(c_i, c_j)
            output['loss_barlow'] = loss_barlow
        if 'inference' in self.output_type:
            output['Cluste'] = torch.argmax(c_i, 1)
            output['probability_Ci'] = c_i
            output['probability_Cj'] = c_j
            output['instance_projector_i'] = z_i
            output['instance_projector_j'] = z_j
            output['res_i'] = h_i
            output['res_j'] = h_j
        return output
