import torch
import torch.nn as nn


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

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 24)
        self.sig = nn.Sigmoid()

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

        # output = self.avgpool(out4)
        # output = torch.flatten(out4, 1)

        output = self.avgpool(out4)
        # output = torch.flatten(output)
        output = torch.squeeze(output, -1)
        output = self.fc(output)

        output = self.sig(output)

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


        ## Skip Connection 1st Block with Addition
        # self.skip_1 = nn.Conv1d(64, 64, kernel_size=(1,), stride=(1,), padding='same', bias=False)

        ## 1st Output
        # self.out_bn_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 2nd Block Beg ##########################################

        self.conv2_1 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn2_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 2nd Block with Addition
        # self.skip_2 = nn.Conv1d(64, 128, kernel_size=(1,), stride=(2,), bias=False)

        ## 2nd Output
        # self.out_bn_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


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

        ## Skip Connection 3rd Block with Addition
        # self.skip_4 = nn.Conv1d(128, 128, kernel_size=(1,), stride=(2,), bias=False)
        #
        # 4th Output
        # self.out_bn_4 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

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
        self.fc_normal = nn.Linear(512, 24)
        self.sig = nn.Sigmoid()

    def _forward_impl(self, x):
        ### 1st Block ###

        conv1_1 = self.conv1_1(x)

        # ext_1 = self.ext_block1(conv1_1)   # 1st Extraction
        # ext_1 = self.relu(ext_1)

        conv1_1 = self.bn1_1(conv1_1)
        conv1_1 = self.relu(conv1_1)

        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.bn1_2(conv1_2)
        conv1_2 = self.relu(conv1_2)

        conv1_3 = self.conv1_3(conv1_2)

        # ext_2 = self.ext_block2(conv1_3)   # 2nd Extraction
        # ext_2 = self.relu(ext_2)

        conv1_3 = self.bn1_3(conv1_3)
        conv1_3 = self.relu(conv1_3)

        ## Skip Connection 1
        # skip_1 = self.skip_1(conv1_1)

        out1 = conv1_1 + conv1_3
        # out1 = self.out_bn_1(out1)
        out1 = self.relu(out1)


        ### 2nd Block ###
        conv2_1 = self.conv2_1(out1)

        # ext_3 = self.ext_block3(conv2_1)  # 3rd Extraction
        # ext_3 = self.relu(ext_3)

        conv2_1 = self.bn2_1(conv2_1)
        conv2_1 = self.relu(conv2_1)

        conv2_2 = self.conv2_2(conv2_1)

        # ext_4 = self.ext_block4(conv2_2)  # 4th Extraction
        # ext_4 = self.relu(ext_4)

        conv2_2 = self.bn2_2(conv2_2)
        conv2_2 = self.relu(conv2_2)

        ## Skip Connection 2
        # skip_2 = self.skip_2(out1)

        out2 = out1 + conv2_2
        # out2 = self.out_bn_2(out2)
        out2 = self.relu(out2)


        ### 3rd Block ###
        conv3_1 = self.conv3_1(out2)

        # ext_5 = self.ext_block5(conv3_1)  # 5th Extraction
        # ext_5 = self.relu(ext_5)

        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)

        conv3_2 = self.conv3_2(conv3_1)

        # ext_6 = self.ext_block6(conv3_2)  # 6th Extraction
        # ext_6 = self.relu(ext_6)

        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)

        ## Skip Connection 3
        skip3 = self.skip_3(out2)

        out3 = skip3 + conv3_2
        out3 = self.out_bn_3(out3)
        out3 = self.relu(out3)

        # ext_skip = self.ext_skip(out3) # skip extraction

        ### 4th Block ###
        conv4_1 = self.conv4_1(out3)
        conv4_1 = self.bn4_1(conv4_1)
        conv4_1 = self.relu(conv4_1)

        conv4_2 = self.conv4_2(conv4_1)

        # ext_7 = self.ext_block7(conv4_2)  # 7th Extraction
        # ext_7 = self.relu(ext_7)

        conv4_2 = self.bn4_2(conv4_2)
        conv4_2 = self.relu(conv4_2)

        ## Skip Connection 4
        # skip4 = self.skip_4(out3)

        out4 = out3 + conv4_2
        # out4 = self.out_bn_4(out4)
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
        # output = torch.flatten(output)
        output = torch.squeeze(output, -1)
        output = self.fc_normal(output)
        output = self.sig(output)

        return output

    def forward(self, x):
        return self._forward_impl(x)
