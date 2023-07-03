import torch
import torch.nn as nn


class ConcentrationPipeline(nn.Module):
    """Process intermediate conv outputs and compress them.
    Parameters
    ----------
    in_filters: int
            Size of the intermediate conv layer filters.
    in_img_size: int
            Size of the extracted input image.
    p_comp: int
            Compression parameter p.
    p_drop : float
            Feature map dropout probability between 0 and 1.
    pool_size: int
            Max pooling kernel size.

    Attributes
    ----------
    compression_out_channels : int
            Output channels of 1x1 convolution.
    comp : nn.Conv1d
            Convolutional layer that compresses the intermediate conv output.
    drop : nn.Dropout1d
            2D Dropout layer for the concentration pipeline.
    max : nn.AvgPool1d
            1D Gap pooling layer.
    flat : nn.Flatten
            Flatten layer for the concatenation.
    """
    def __init__(self, in_filters, p_comp, pool_size, p_drop=0.3):
        super().__init__()
        self.in_filters = in_filters
        # self.wsize = wsize
        self.p_comp = p_comp
        self.pool_size = pool_size
        self.p_drop = p_drop

        self.compression_out_channels = in_filters // p_comp

        self.comp = nn.Conv2d(in_filters, self.compression_out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.compression_out_channels, eps=1e-05, momentum=0.1,
                                 affine=True, track_running_stats=True)
        self.drop = nn.Dropout(p_drop)
        self.gap = nn.AvgPool2d(pool_size)
        self.flat = nn.Flatten()

    def forward(self, x):
        """Run Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(batch_size, flat_dim)`.
        """
        x = self.comp(
            x
        )                       # (batch_size, compression_out_channels, w_size)
        x = self.bn(x)
        x = self.drop(x)        # (batch_size, compression_out_channels, w_size)
        x = self.gap(x)         # (batch_size, compression_out_channels, w_size//pool_size, w_size//pool_size)
        x = self.flat(x)

        return x


class ExtractionBlock(nn.Module):
    """Create an arbitrary number of extraction pipelines.
    Parameters
    ----------
    in_filters: int
            Size of the intermediate conv layer filters.
    in_img_size: int
            Size of the extracted input image.
    p_comp: int
            Compression parameter p.
    p_drop : float
            Feature map dropout probability between 0 and 1.
    pool_sizes: list
            Max pooling kernel size.

    Attributes
    ----------
    compression_out_channels : int
            Output channels of 1x1 convolution.
    comp : nn.Conv1d
            Convolutional layer that compresses the intermediate conv output.
    drop : nn.Dropout
            1D Dropout layer for the concentration pipeline.
    gap : nn.AvgPool1D
            1D Avg pooling layer.
    flat : nn.Flatten
            Flatten layer for the concatenation.
    """

    def __init__(self, in_filters,  p_comp, pool_size, p_drop):
        super().__init__()
        pipelines = []
        for i in range(len(pool_size)):
            pipeline = ConcentrationPipeline(
                in_filters,
                p_comp,
                pool_size[i],
                p_drop
            )
            compression_out_channels = in_filters // p_comp

            pipelines.append(pipeline)
            # mlps.append(mlp)

        self.pipelines = nn.ModuleList(pipelines)
        # self.mlps = nn.ModuleList(mlps)

    def forward(self, x):
        """Run Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        List of torch.Tensors
            Shape `(batch_size, flat_dim)`.
        """
        # x_init = x.detach().clone().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        x_init = x.clone()
        out = []
        for pipe in self.pipelines:
            x = pipe(x_init)
            out.append(x)
        out = torch.cat(out, 1)
        return out


class BioDG(nn.Module):
    def __init__(self, channels: int):
        super(BioDG, self).__init__()

        ########################################## 1st Block Beg ##########################################

        self.conv1_1 = nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn1_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ## Skip Connection 1st Block with Addition
        self.skip_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same', bias=False)

        ## 1st Output
        self.out_bn_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 2nd Block Beg ##########################################

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 2nd Block with Addition
        self.skip_2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)

        ## 2nd Output
        self.out_bn_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 3rd Block Beg ##########################################

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 3rd Block with Addition
        self.skip_3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)

        ## 3rd Output
        self.out_bn_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 4th Block Beg ##########################################

        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Skip Connection 3rd Block with Addition
        self.skip_4 = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False)

        # 4th Output
        self.out_bn_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn1_3.weight, 0)
        nn.init.constant_(self.bn2_2.weight, 0)
        nn.init.constant_(self.bn3_2.weight, 0)
        nn.init.constant_(self.bn4_2.weight, 0)


        ###### Extraction Blocks ######

        self.ext_block1 = ExtractionBlock(
            in_filters=64,
            p_comp=4,
            pool_size=[3],
            p_drop=0.3
        )

        self.ext_block2 = ExtractionBlock(
            in_filters=64,
            p_comp=4,
            pool_size=[3],
            p_drop=0.5
        )

        self.ext_block3 = ExtractionBlock(
            in_filters=128,
            p_comp=4,
            pool_size=[1],
            p_drop=0.5
        )

        self.ext_block4 = ExtractionBlock(
            in_filters=128,
            p_comp=4,
            pool_size=[1],
            p_drop=0.5
        )

        self.ext_block5 = ExtractionBlock(
            in_filters=256,
            p_comp=4,
            pool_size=[1],
            p_drop=0.5
        )

        self.ext_block6 = ExtractionBlock(
            in_filters=256,
            p_comp=4,
            pool_size=[1],
            p_drop=0.5
        )

        self.ext_block7 = ExtractionBlock(
            in_filters=256,
            p_comp=4,
            pool_size=[1],
            p_drop=0.5
        )

        self.ext_skip = ExtractionBlock(
            in_filters=256,
            p_comp=4,
            pool_size=[1],
            p_drop=0.3
        )

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(65280, 3)
        self.fc = nn.Linear(41664, 3)
        # self.sig = nn.Sigmoid()

    def _forward_impl(self, x):
        ### 1st Block ###

        conv1_1 = self.conv1_1(x)

        ext1 = self.ext_block1(conv1_1)       # 1st Extraction
        ext1 = self.relu(ext1)

        conv1_1 = self.bn1_1(conv1_1)
        conv1_1 = self.relu(conv1_1)

        conv1_2 = self.conv1_2(conv1_1)
        # ext2 = self.ext_block2(conv1_2)
        # ext2 = self.relu(ext2)

        conv1_2 = self.bn1_2(conv1_2)
        conv1_2 = self.relu(conv1_2)

        conv1_3 = self.conv1_3(conv1_2)

        ext2 = self.ext_block2(conv1_3)       # 2nd Extraction
        ext2 = self.relu(ext2)

        conv1_3 = self.bn1_3(conv1_3)
        conv1_3 = self.relu(conv1_3)

        ## Skip Connection 1
        skip_1 = self.skip_1(conv1_1)

        out1 = skip_1 + conv1_3
        out1 = self.out_bn_1(out1)
        out1 = self.relu(out1)


        ### 2nd Block ###
        conv2_1 = self.conv2_1(out1)

        ext3 = self.ext_block3(conv2_1)    # 3rd Extraction
        ext3 = self.relu(ext3)

        conv2_1 = self.bn2_1(conv2_1)
        conv2_1 = self.relu(conv2_1)

        conv2_2 = self.conv2_2(conv2_1)

        ext4 = self.ext_block4(conv2_2)    # 4th Extraction
        ext4 = self.relu(ext4)

        conv2_2 = self.bn2_2(conv2_2)
        conv2_2 = self.relu(conv2_2)

        ## Skip Connection 2
        skip_2 = self.skip_2(out1)

        out2 = skip_2 + conv2_2
        out2 = self.out_bn_2(out2)
        out2 = self.relu(out2)


        ### 3rd Block ###
        conv3_1 = self.conv3_1(out2)

        ext5 = self.ext_block6(conv3_1)   # 5th Extraction
        ext5 = self.relu(ext5)

        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)

        conv3_2 = self.conv3_2(conv3_1)

        ext6 = self.ext_block6(conv3_2)
        ext6 = self.relu(ext6)

        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)

        ## Skip Connection 3
        skip3 = self.skip_3(out2)

        out3 = skip3 + conv3_2
        out3 = self.out_bn_3(out3)
        out3 = self.relu(out3)

        ext_skip = self.ext_skip(out3)

        ### 4th Block ###
        conv4_1 = self.conv4_1(out3)
        conv4_1 = self.bn4_1(conv4_1)
        conv4_1 = self.relu(conv4_1)

        conv4_2 = self.conv4_2(conv4_1)

        ext7 = self.ext_block7(conv4_2)  # 7th Extraction
        ext7 = self.relu(ext7)

        conv4_2 = self.bn4_2(conv4_2)
        conv4_2 = self.relu(conv4_2)

        ## Skip Connection 4
        skip4 = self.skip_4(out3)

        out4 = skip4 + conv4_2
        out4 = self.out_bn_4(out4)
        out4 = self.relu(out4)

        # output = self.avgpool(out4)
        output = torch.flatten(out4, 1)

        output = torch.cat([ext1, ext2, ext3, ext4,
                            ext5, ext6, ext_skip, ext7, output], 1)

        # output = self.fc(output)

        # output = self.sig(output)

        return output

    def forward(self, x):
        return self._forward_impl(x)


class SResNet(nn.Module):
    def __init__(self, channels: int):
        super(SResNet, self).__init__()

        ########################################## 1st Block Beg ##########################################

        self.conv1_1 = nn.Conv2d(channels, 128, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn1_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1_3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding='same', bias=False)
        self.bn1_3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ## Skip Connection 1st Block with Addition
        self.skip_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding='same', bias=False)

        ## 1st Output
        self.out_bn_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        ########################################## 2nd Block Beg ##########################################

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        ## Skip Connection 2nd Block with Addition
        self.skip_2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)

        ## 2nd Output
        self.out_bn_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        # ########################################## 3rd Block Beg ##########################################
        #
        # self.conv3_1 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        # self.bn3_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #
        # self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        # self.bn3_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #
        # ## Skip Connection 3rd Block with Addition
        # self.skip_3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        #
        # ## 3rd Output
        # self.out_bn_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #
        #
        # ########################################## 4th Block Beg ##########################################
        #
        # self.conv4_1 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2, bias=False)
        # self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        # self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #
        # # Skip Connection 3rd Block with Addition
        # self.skip_4 = nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False)
        #
        # # 4th Output
        # self.out_bn_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.bn1_3.weight, 0)
        nn.init.constant_(self.bn2_2.weight, 0)
        # nn.init.constant_(self.bn3_2.weight, 0)
        # nn.init.constant_(self.bn4_2.weight, 0)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(65280, 3)
        # self.sig = nn.Sigmoid()

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


        # ### 3rd Block ###
        # conv3_1 = self.conv3_1(out2)
        # conv3_1 = self.bn3_1(conv3_1)
        # conv3_1 = self.relu(conv3_1)
        #
        # conv3_2 = self.conv3_2(conv3_1)
        # conv3_2 = self.bn3_2(conv3_2)
        # conv3_2 = self.relu(conv3_2)
        #
        # ## Skip Connection 3
        # skip3 = self.skip_3(out2)
        #
        # out3 = skip3 + conv3_2
        # out3 = self.out_bn_3(out3)
        # out3 = self.relu(out3)
        #
        #
        # ### 4th Block ###
        # conv4_1 = self.conv4_1(out3)
        # conv4_1 = self.bn4_1(conv4_1)
        # conv4_1 = self.relu(conv4_1)
        #
        # conv4_2 = self.conv4_2(conv4_1)
        # conv4_2 = self.bn4_2(conv4_2)
        # conv4_2 = self.relu(conv4_2)
        #
        # ## Skip Connection 4
        # skip4 = self.skip_4(out3)
        #
        # out4 = skip4 + conv4_2
        # out4 = self.out_bn_4(out4)
        # out4 = self.relu(out4)

        # output = self.avgpool(out4)
        output = torch.flatten(out2, 1)
        output = self.fc(output)

        # output = self.sig(output)

        return output

    def forward(self, x):
        return self._forward_impl(x)
