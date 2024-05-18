import torch.nn as nn
import math
from models.yh_const import ANGLE_CLASSES_NUM, CLASS_NUM, REGRESS_NUM
import torch.nn.functional as F

class vanilla_conv2d(nn.Module):
    def __init__(self, windowSize):
        super(vanilla_conv2d, self).__init__()   #tactile 64*64
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        # 32*64*64
        # 32*64*64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 64*32*32
        # 64*16*16

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        #128*32*32
        #128*16*16

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24
        #256*16*16
        #256*8*8

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        #512*16*16
        #512*8*8

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))
        #1024*12*12
        #1024*8*8

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2)) # 10 * 10
        #1024*6*6
        #1024*3*3

        self.flatten = nn.Flatten()
        self.feature_size = 1024*6*6


        self.regression = nn.Sequential(
            nn.Linear(self.feature_size, 512), #for 32*32: nn.Linear(1024*2*2, 512)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, REGRESS_NUM),
        )

        self.classification = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, CLASS_NUM),
            nn.Softmax()
        )


    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = self.flatten(output)
        values = self.regression(output)
        classes = self.classification(output)
        return values, classes

class vanilla_conv3d(nn.Module):
    def __init__(self, window_size, class_num, regress_num):
        super(vanilla_conv3d, self).__init__()

        # tactile 1*n*64*64, n = window size

        self.conv_0 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32))
        # 32*n*64*64

        self.conv_1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ) # 48 * 48
        # 64*n*32*32

        self.conv_2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))
        # 128*n*32*32

        self.conv_3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ) # 24 * 24
        # 256*n*16*16

        self.conv_4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512))
        # 512*n*16*16

        self.conv_5 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=5),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1024))
        # 1024*n-4*12*12

        self.conv_6 = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1024),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ) # 10 * 10
        # 1024*(n-4)/2*6*6

        new_n = math.ceil((window_size-4)/2)

        self.dense_0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*new_n*6*6, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.feature_size = 1024

        self.classification = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, CLASS_NUM),
            nn.Softmax()
        )

    def forward(self, input):
        if len(input.size()) == 4:
            input = input.unsqueeze(1)
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = self.dense_0(output)
        #values = self.regression(output)
        classes = self.classification(output)
        return classes


class vanilla_conv2d_v2(nn.Module):
    def __init__(self, windowSize):
        super(vanilla_conv2d_v2, self).__init__()   #tactile 64*64
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        # 32*64*64
        # 32*32*32

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 64*32*32
        # 64*16*16

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        #128*32*32
        #128*16*16

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24
        #256*16*16
        #256*8*8

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        #512*16*16
        #512*8*8

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))
        #1024*12*12
        #1024*4*4

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2)) # 10 * 10
        #1024*6*6
        #1024*2*2

        self.flatten = nn.Flatten()
        # self.feature_size = 1024*6*6
        self.feature_size = 1024 * 2 * 2

        self.speed = nn.Sequential(
            nn.Linear(self.feature_size, 512), #for 32*32: nn.Linear(1024*2*2, 512)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

        self.motion = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, CLASS_NUM),
            nn.Softmax()
        )

        self.angle = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, ANGLE_CLASSES_NUM),
            nn.Softmax()
        )

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = self.flatten(output)
        return self.motion(output)


cfg = {
    'VGG11': [32, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [32, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 20
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2))
        # 4,64,64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 8*32*32


        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2))
        #16*16*16


        self.conv_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24
        #32*8*8

        self.flatten = nn.Flatten()

        self.conv_4 = nn.Sequential(
            nn.Linear(1024, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )


        # self.conv_4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(512))
        # #512*16*16
        # #512*8*8
        #
        # self.conv_5 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=(5,5)),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(1024))

    def forward(self, input):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.flatten(output)
        output = self.conv_4(output)
        # output = self.conv_4(output)
        # output = self.conv_5(output)
        # output = self.conv_6(output)
        output = output.view(-1, 256)
        return output


class LSTM_CNN(nn.Module):
    def __init__(self):
        super(LSTM_CNN, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, CLASS_NUM),
            nn.Softmax()
        )

    def forward(self, input):
        batch_size, timestamps, C, H, W = input.size()
        cnn_in = input.view(batch_size * timestamps, C, H, W)
        cnn_out = self.cnn(cnn_in)
        rnn_in = cnn_out.view(batch_size, timestamps, -1)
        rnn_out, (h_n, h_c) = self.rnn(rnn_in)
        rnn_out2 = self.linear(rnn_out[:,-1,:])

        return rnn_out2
