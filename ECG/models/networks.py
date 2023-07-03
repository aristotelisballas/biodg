# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from ECG.lib import wide_resnet
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    elif input_shape == (12, 5000):
        if hparams['resnet18']:
            return ResNet18()
        else:
            return SResNet()
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class SResNet(nn.Module):
    def __init__(self, ):
        super(SResNet, self).__init__()

        ########################################## 1st Block Beg ##########################################

        self.conv1_1 = nn.Conv1d(12, 64, kernel_size=(8,), stride=(1,), padding='same', bias=False)
        self.bn1_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding='same', bias=False)
        self.bn1_2 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_3 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn1_3 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ## Skip Connection 1st Block with Addition
        self.skip_1 = nn.Conv1d(64, 64, kernel_size=(1,), stride=(1,), padding='same', bias=False)

        ## 1st Output
        self.out_bn_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 2nd Block Beg ##########################################

        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=(8,), stride=(2,), padding=(3,), bias=False)
        self.bn2_1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=(5,), stride=(1,), padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 2nd Block with Addition
        self.skip_2 = nn.Conv1d(64, 128, kernel_size=(1,), stride=(2,), bias=False)

        ## 2nd Output
        self.out_bn_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 3rd Block Beg ##########################################

        self.conv3_1 = nn.Conv1d(128, 256, kernel_size=(8,), stride=(2,), padding=(3,), bias=False)
        self.bn3_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_2 = nn.Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding='same', bias=False)
        self.bn3_2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 3rd Block with Addition
        self.skip_3 = nn.Conv1d(128, 256, kernel_size=(1,), stride=(2,), bias=False)

        ## 3rd Output
        self.out_bn_3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 4th Block Beg ##########################################

        self.conv4_1 = nn.Conv1d(256, 256, kernel_size=(8,), stride=(2,), padding=(3,), bias=False)
        self.bn4_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_2 = nn.Conv1d(256, 256, kernel_size=(5,), stride=(1,), padding='same', bias=False)
        self.bn4_2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 3rd Block with Addition
        self.skip_4 = nn.Conv1d(256, 256, kernel_size=(1,), stride=(2,), bias=False)

        ## 4th Output
        self.out_bn_4 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn1_3.weight, 0)
        nn.init.constant_(self.bn2_2.weight, 0)
        nn.init.constant_(self.bn3_2.weight, 0)
        nn.init.constant_(self.bn4_2.weight, 0)

    def _forward_impl(self, x):
        ### 1st Block ###

        conv1_1 = self.conv1_1(x)
        conv1_1 = self.bn1_1(conv1_1)
        conv1_1 = self.relu(conv1_1)

        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.bn1_2(conv1_2)
        conv1_2 = self.relu(conv1_2)

        conv1_3 = self.conv1_3(conv1_2)
        conv1_3 = self.bn1_3(conv1_3)
        conv1_3 = self.relu(conv1_3)

        ## Skip Connection 1
        skip_1 = self.skip_1(conv1_1)

        out1 = skip_1 + conv1_3
        out1 = self.out_bn_1(out1)
        out1 = self.relu(out1)


        ### 2nd Block ###
        conv2_1 = self.conv2_1(out1)
        conv2_1 = self.bn2_1(conv2_1)
        conv2_1 = self.relu(conv2_1)

        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = self.bn2_2(conv2_2)
        conv2_2 = self.relu(conv2_2)

        ## Skip Connection 2
        skip_2 = self.skip_2(out1)

        out2 = skip_2 + conv2_2
        out2 = self.out_bn_2(out2)
        out2 = self.relu(out2)


        ### 3rd Block ###
        conv3_1 = self.conv3_1(out2)
        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)

        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)

        ## Skip Connection 3
        skip3 = self.skip_3(out2)

        out3 = skip3 + conv3_2
        out3 = self.out_bn_3(out3)
        out3 = self.relu(out3)


        ### 4th Block ###
        conv4_1 = self.conv4_1(out3)
        conv4_1 = self.bn4_1(conv4_1)
        conv4_1 = self.relu(conv4_1)

        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.bn4_2(conv4_2)
        conv4_2 = self.relu(conv4_2)

        ## Skip Connection 4
        skip4 = self.skip_4(out3)

        out4 = skip4 + conv4_2
        out4 = self.out_bn_4(out4)
        out4 = self.relu(out4)

        output = torch.flatten(out4, 1)

        return output

    def forward(self, x):
        return self._forward_impl(x)


class ResNet18(nn.Module):
    def __init__(self, p=4):
        super(ResNet18, self).__init__()
        self.p = p

        self.conv1_1 = nn.Conv1d(12, 64, kernel_size=(7,), stride=(1,), padding='same', bias=False)
        self.bn1_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        ########################################## 1st Block Beg ##########################################
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn1_2 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_3 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn1_3 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 2nd Block Beg ##########################################

        self.conv2_1 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn2_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)



        ########################################## 3rd Block Beg ##########################################

        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn3_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 3rd Block with Addition
        self.skip_3 = nn.Conv1d(64, 128, kernel_size=(1,), stride=(2,), bias=False)

        ## 3rd Output
        self.out_bn_3 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 4th Block Beg ##########################################

        self.conv4_1 = nn.Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn4_1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_2 = nn.Conv1d(128, 128, kernel_size=(5,), stride=(1,), padding='same', bias=False)
        self.bn4_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 5th Block Beg ##########################################
        self.conv5_1 = nn.Conv1d(128, 256, kernel_size=(3,), stride=2, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_2 = nn.Conv1d(256, 256, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn5_2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Skip Connection
        self.skip_5 = nn.Conv1d(128, 256, kernel_size=(1,), stride=(2,), bias=False)

        # 5th output
        self.out_bn_5 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ########################################## 6th Block Beg ##########################################
        self.conv6_1 = nn.Conv1d(256, 256, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn6_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv6_2 = nn.Conv1d(256, 256, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn6_2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ########################################## 7th Block Beg ##########################################
        self.conv7_1 = nn.Conv1d(256, 512, kernel_size=(3,), stride=2, padding=1, bias=False)
        self.bn7_1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv7_2 = nn.Conv1d(512, 512, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn7_2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Skip Connection
        self.skip_7 = nn.Conv1d(256, 512, kernel_size=(1,), stride=(2,), bias=False)

        # 7th output
        self.out_bn_7 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 7th Block Beg ##########################################
        self.conv8_1 = nn.Conv1d(512, 512, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn8_1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv8_2 = nn.Conv1d(512, 512, kernel_size=(3,), stride=1, padding='same', bias=False)
        self.bn8_2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn1_3.weight, 0)
        nn.init.constant_(self.bn2_2.weight, 0)
        nn.init.constant_(self.bn3_2.weight, 0)
        nn.init.constant_(self.bn4_2.weight, 0)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _forward_impl(self, x):
        ### 1st Block ###

        conv1_1 = self.conv1_1(x)
        conv1_1 = self.bn1_1(conv1_1)
        conv1_1 = self.relu(conv1_1)

        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.bn1_2(conv1_2)
        conv1_2 = self.relu(conv1_2)

        conv1_3 = self.conv1_3(conv1_2)
        conv1_3 = self.bn1_3(conv1_3)
        conv1_3 = self.relu(conv1_3)

        ## Skip Connection 1
        out1 = conv1_1 + conv1_3
        out1 = self.relu(out1)


        ### 2nd Block ###
        conv2_1 = self.conv2_1(out1)
        conv2_1 = self.bn2_1(conv2_1)
        conv2_1 = self.relu(conv2_1)

        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = self.bn2_2(conv2_2)
        conv2_2 = self.relu(conv2_2)

        ## Skip Connection 2
        out2 = out1 + conv2_2
        out2 = self.relu(out2)


        ### 3rd Block ###
        conv3_1 = self.conv3_1(out2)
        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)

        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)

        ## Skip Connection 3
        skip3 = self.skip_3(out2)

        out3 = skip3 + conv3_2
        out3 = self.out_bn_3(out3)
        out3 = self.relu(out3)

        ### 4th Block ###
        conv4_1 = self.conv4_1(out3)
        conv4_1 = self.bn4_1(conv4_1)
        conv4_1 = self.relu(conv4_1)

        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.bn4_2(conv4_2)
        conv4_2 = self.relu(conv4_2)

        ## Skip Connection 4
        out4 = out3 + conv4_2
        out4 = self.relu(out4)

        ### 5th block ###
        conv5_1 = self.conv5_1(out4)
        conv5_1 = self.bn5_1(conv5_1)
        conv5_1 = self.relu(conv5_1)

        conv5_2 = self.conv5_2(conv5_1)
        conv5_2 = self.bn5_2(conv5_2)
        conv5_2 = self.relu(conv5_2)

        ## skip 5
        skip5 = self.skip_5(out4)

        out5 = skip5 + conv5_2
        out5 = self.out_bn_5(out5)
        out5 = self.relu(out5)

        ### 6th block ###
        conv6_1 = self.conv6_1(out5)
        conv6_1 = self.bn6_1(conv6_1)
        conv6_1 = self.relu(conv6_1)

        conv6_2 = self.conv6_2(conv6_1)
        conv6_2 = self.bn6_2(conv6_2)
        conv6_2 = self.relu(conv6_2)

        out6 = out5 + conv6_2
        out6 = self.relu(out6)

        ### 7th Block
        conv7_1 = self.conv7_1(out6)
        conv7_1 = self.bn7_1(conv7_1)
        conv7_1 = self.relu(conv7_1)

        conv7_2 = self.conv7_2(conv7_1)
        conv7_2 = self.bn7_2(conv7_2)
        conv7_2 = self.relu(conv7_2)

        ## skip 7
        skip7 = self.skip_7(out6)

        out7 = skip7 + conv7_2
        out7 = self.out_bn_7(out7)
        out7 = self.relu(out7)

        ### 8th block
        conv8_1 = self.conv8_1(out7)
        conv8_1 = self.bn8_1(conv8_1)
        conv8_1 = self.relu(conv8_1)

        conv8_2 = self.conv8_2(conv8_1)
        conv8_2 = self.bn8_2(conv8_2)
        conv8_2 = self.relu(conv8_2)

        out8 = out7 + conv8_2
        out8 = self.relu(out8)

        output = self.avgpool(out8)
        output = torch.squeeze(output, -1)

        return output

    def forward(self, x):
        return self._forward_impl(x)
