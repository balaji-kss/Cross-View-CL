import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from modelZoo.i3dpt import I3D, I3D_head


class BaseNet(nn.Module):
    """
    Backbone network of the model
    """

    def __init__(self, base_name, data_type, kinetics_pretrain):

        super(BaseNet, self).__init__()

        self.base_name = base_name
        # self.kinetics_pretrain = cfg.kinetics_pretrain
        self.kinetics_pretrain = kinetics_pretrain
        self.freeze_stats = True
        self.freeze_affine = True
        self.fp16 = False
        self.data_type = data_type

        if self.base_name == "i3d":
            self.base_model = build_base_i3d(self.data_type, self.kinetics_pretrain, self.freeze_affine)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Applies network layers on input images

        Args:
            x: input image sequences. Shape: [batch_size, T, C, W, H]
        """

        x = x.permute(0, 2, 1, 3, 4)  # [N,T,C,W,H] --> [N,C,T,W,H]
        conv_feat = self.base_model(x)

        # reshape to original size
        conv_feat = conv_feat.permute(0, 2, 1, 3, 4)  # [N,C,T,W,H] --> [N,T,C,W,H]

        return conv_feat


def build_base_i3d(data_type, kinetics_pretrain=None, freeze_affine=True):
    # print("Building I3D model...")

    i3d = I3D(num_classes=400, data_type=data_type)
    # kinetics_pretrain = '/pretrained/i3d_flow_kinetics.pth'
    if kinetics_pretrain is not None:
        if os.path.isfile(kinetics_pretrain):
            # print("Loading I3D pretrained on Kinetics dataset from {}...".format(kinetics_pretrain))
            print('Loading pretrained I3D:')
            i3d.load_state_dict(torch.load(kinetics_pretrain))
        else:
            raise ValueError("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))

    base_model = nn.Sequential(i3d.conv3d_1a_7x7,
                               i3d.maxPool3d_2a_3x3,
                               i3d.conv3d_2b_1x1,
                               i3d.conv3d_2c_3x3,
                               i3d.maxPool3d_3a_3x3,
                               i3d.mixed_3b,
                               i3d.mixed_3c,
                               i3d.maxPool3d_4a_3x3,
                               i3d.mixed_4b,
                               i3d.mixed_4c,
                               i3d.mixed_4d,
                               i3d.mixed_4e,
                               i3d.mixed_4f)

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad = False

    if freeze_affine:
        base_model.apply(set_bn_fix)


    for p in base_model.parameters():
        p.requires_grad = False

    return base_model

def build_conv(base_name='i3d', kinetics_pretrain=None, mode='global', freeze_affine=True):

    if base_name == "i3d":

        i3d = I3D_head()

        model_dict = i3d.state_dict()
        if kinetics_pretrain is not None:
            if os.path.isfile(kinetics_pretrain):
                # print ("Loading I3D head pretrained")
                pretrained_dict = torch.load(kinetics_pretrain)
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                i3d.load_state_dict(model_dict)
            else:
                raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))
        #

        model = nn.Sequential(i3d.maxPool3d,
                                  i3d.mixed_5b,
                                  i3d.mixed_5c,
                              i3d.avg_pool)
        # else:
        # #     # for global branch
        #     model = nn.Sequential(i3d.mixed_5b,
        #                        i3d.mixed_5c)

    else:
        raise NotImplementedError

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        model.apply(set_bn_fix)

    return model


class RGBAction(nn.Module):
    def __init__(self, num_class, kinetics_pretrain):

        super(RGBAction, self).__init__()
        self.num_class = num_class
        self.base_net = 'i3d'
        self.data_type = 'rgb'
        self.freeze_stats = True
        self.freeze_affine = True
        self.fc_dim = 256
        self.dropout_prob = 0.3
        self.temp_size = 8 #for T=36
        self.fp16 = False
        self.kinetics_pretrain = kinetics_pretrain

        self.I3D_head = BaseNet(self.base_net, self.data_type, self.kinetics_pretrain)
        self.Global = build_conv(self.base_net, self.kinetics_pretrain, 'global', self.freeze_affine)

        # self.Context = build_conv(self.base_net, self.kinetics_pretrain, 'context', self.freeze_affine)
        self.cat = nn.Conv3d(832+832, 832, kernel_size=1, stride=1, bias=True)
        self.layer1 = nn.Conv3d(1024, self.fc_dim,
                                    kernel_size=1, stride=1, bias=True)

        # self.global_cls = nn.Conv3d(
        #         self.fc_dim * self.pool_size**2,
        #         self.num_class,
        #         (1,1,1),
        #         bias=True)

        # self.global_cls = nn.Conv3d(self.fc_dim, self.num_class,(1,1,1), bias=True )
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(self.fc_dim*self.temp_size, 512)
        self.bn = nn.BatchNorm1d(512)

        self.cls = nn.Linear(512, self.num_class)


        self.dropout = nn.Dropout(self.dropout_prob)


    def forward(self, gl,roi):

        globalFeat= self.I3D_head(gl)
        roiFeat = self.I3D_head(roi)

        N, T, _,_,_ = globalFeat.size()

        concatFeat = torch.cat((globalFeat, roiFeat),2) #chanel-wise concat

        concatFeat = self.cat(concatFeat.permute(0,2,1,3,4))

        STconvFeat = self.Global(concatFeat)

        STconvFeat = self.layer1(STconvFeat)

        STconvFeat_final = self.dropout(STconvFeat)

        STconvFeat_final_flatten = STconvFeat_final.view(N, -1)
        featOut = self.relu(self.bn(self.fc(STconvFeat_final_flatten)))
        out = self.cls(featOut)

        return out, featOut

if __name__ == '__main__':
    gpu_id = 4
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    net = RGBAction(num_class=12, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    globalImage = torch.randn(5, 40, 3, 224, 224).cuda(gpu_id)
    roiImage = torch.randn_like(globalImage)
    pred,_ = net(globalImage, roiImage)

    print(pred.shape)