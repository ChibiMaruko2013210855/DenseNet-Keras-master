"""Test ImageNet pretrained DenseNet"""

import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
import keras
import pylab as plt
from keras.datasets import cifar10
from densenet121 import DenseNet

nb_classes=10
# Test pretrained model
model = DenseNet(reduction=0.5, classes=nb_classes)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 数据的导入和预处理
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
# X_train和X_test是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）
# Y_train和 Y_test是形如（nb_samples,）标签数据，标签的范围是0~9

y_train=keras.utils.to_categorical(y_train,nb_classes)


# 训练模型
model.fit(x=x_train,y=y_train,epochs=30,batch_size=64)
model.save_weights('imagenet_models/densenet121_weights_tf_cifar10.h5')





