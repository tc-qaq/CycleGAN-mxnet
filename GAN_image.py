from mxnet import gluon, image, nd
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata, nn
from skimage import io
import mxnet as mx
import numpy as np
import os

class conv_inst_relu(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(conv_inst_relu, self).__init__()
        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
                self.net.add(
                    nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2),
                    nn.InstanceNorm(),
                    nn.Activation('relu')

                )

    def hybrid_forward(self, F, x):
        return self.net(x)

class upconv(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(upconv, self).__init__()
        self.conv = nn.Conv2D(filters, kernel_size=3, padding=1,strides=1)

    def hybrid_forward(self, F, x):
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        return self.conv(x)

class upconv_inst_relu(gluon.nn.HybridBlock):
    def __init__(self, filters):
        super(upconv_inst_relu, self).__init__()

        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
                self.net.add(
                    upconv(filters),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )

    def hybrid_forward(self, F, x):
        return self.net(x)

class deconv_bn_relu(gluon.nn.HybridBlock):
    def __init__(self, NumLayer, filters):
        super(deconv_bn_relu, self).__init__()
        self.NumLayer = NumLayer
        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for i in range(NumLayer-1):
                self.net.add(
                    nn.Conv2DTranspose(self.filters,kernel_size=4, padding=1, strides=2),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )
        self.net.add(
            nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
        )

    def hybrid_forward(self, F, x):
        return self.net(x)

class ResBlock(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(ResBlock, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(filters, kernel_size=3, padding=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1),
                nn.InstanceNorm(),
                nn.Activation('relu')
            )
    def hybrid_forward(self, F, x):
        out = self.net(x)
        return out + x

class Generator_256(gluon.nn.HybridBlock):
    def __init__(self):
        super(Generator_256, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(32, kernel_size=7, strides=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),  #c7s1-32
                conv_inst_relu(64),
                conv_inst_relu(128),
            )
            for _ in range(9):
                self.net.add(
                        ResBlock(128)
                )
            self.net.add(
                upconv_inst_relu(64),
                upconv_inst_relu(32),
                nn.ReflectionPad2D(3),
                nn.Conv2D(3,kernel_size=7,strides=1),
                nn.Activation('sigmoid')
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class Generator_128(gluon.nn.HybridBlock):
    def __init__(self):
        super(Generator_128, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(32, kernel_size=7, strides=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),  #c7s1-32
                conv_inst_relu(64),
                conv_inst_relu(128),
            )
            for _ in range(6):
                self.net.add(
                        ResBlock(128)
                )
            self.net.add(
                upconv_inst_relu(64),
                upconv_inst_relu(32),
                nn.ReflectionPad2D(3),
                nn.Conv2D(3,kernel_size=7,strides=1),
                nn.Activation('sigmoid')
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class Discriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(64, kernel_size=3,strides=2,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2D(128, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(256, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(512, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(1,kernel_size=1,strides=1),
            )
    def  hybrid_forward(self, F, x):
        return self.net(x)




