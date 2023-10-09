# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b/mit_semseg/models/models.py
# https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
	"3x3 convolution + BN + relu"
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True),
	)
# upernet
class UPerDecoder(nn.Module):
	def __init__(self,
	    in_dim=[256, 512, 1024, 2048],
	    ppm_pool_scale=[1, 2, 3, 6],
	    ppm_dim=512,
	    fpn_out_dim=256
	):
		super(UPerDecoder, self).__init__()
		
		# PPM ----
		dim = in_dim[-1]
		ppm_pooling = []
		ppm_conv = []
		
		for scale in ppm_pool_scale:
			ppm_pooling.append(
				nn.AdaptiveAvgPool2d(scale)
			)
			ppm_conv.append(
				nn.Sequential(
					nn.Conv2d(dim, ppm_dim, kernel_size=1, bias=False),
					nn.BatchNorm2d(ppm_dim),
					nn.ReLU(inplace=True)
				)
			)
		self.ppm_pooling   = nn.ModuleList(ppm_pooling)
		self.ppm_conv      = nn.ModuleList(ppm_conv)
		self.ppm_out = conv3x3_bn_relu(dim + len(ppm_pool_scale)*ppm_dim, fpn_out_dim, 1)
		
		# FPN ----
		fpn_in = []
		for i in range(0, len(in_dim)-1):  # skip the top layer
			fpn_in.append(
				nn.Sequential(
					nn.Conv2d(in_dim[i], fpn_out_dim, kernel_size=1, bias=False),
					nn.BatchNorm2d(fpn_out_dim),
					nn.ReLU(inplace=True)
				)
			)
		self.fpn_in = nn.ModuleList(fpn_in)
		
		fpn_out = []
		for i in range(len(in_dim) - 1):  # skip the top layer
			fpn_out.append(
				conv3x3_bn_relu(fpn_out_dim, fpn_out_dim, 1),
			)
		self.fpn_out = nn.ModuleList(fpn_out)
		
		self.fpn_fuse = nn.Sequential(
			conv3x3_bn_relu(len(in_dim) * fpn_out_dim, fpn_out_dim, 1),
		)
	
	def forward(self, feature):
		f = feature[-1]
		pool_shape = f.shape[2:]
		
		ppm_out = [f]
		for pool, conv in zip(self.ppm_pooling, self.ppm_conv):
			p = pool(f)
			p = F.interpolate(p, size=pool_shape, mode='bilinear', align_corners=False)
			p = conv(p)
			ppm_out.append(p)
		ppm_out = torch.cat(ppm_out, 1)
		down = self.ppm_out(ppm_out)
		
		
		#--------------------------------------
		fpn_out = [down]
		for i in reversed(range(len(feature) - 1)):
			lateral = feature[i]
			lateral = self.fpn_in[i](lateral) # lateral branch
			down = F.interpolate(down, size=lateral.shape[2:], mode='bilinear', align_corners=False) # top-down branch
			down = down + lateral
			fpn_out.append(self.fpn_out[i](down))
		
		fpn_out.reverse() # [P2 - P5]
		fusion_shape = fpn_out[0].shape[2:]
		fusion = [fpn_out[0]]
		for i in range(1, len(fpn_out)):
			fusion.append(
				F.interpolate( fpn_out[i], fusion_shape, mode='bilinear', align_corners=False)
			)
		x = self.fpn_fuse( torch.cat(fusion, 1))
		
		return x, fusion
