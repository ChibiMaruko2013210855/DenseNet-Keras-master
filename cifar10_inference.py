import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
import keras
import pylab as plt
from keras.datasets import cifar10
from densenet121 import DenseNet


# 数据的导入和预处理
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
y_test=keras.utils.to_categorical(y_test,10)

# 模型已经训练好了，将训练好的权重加入到模型中
weights_path = 'imagenet_models/densenet121_weights_tf_cifar10.h5'
model = DenseNet(reduction=0.5, classes=10, weights_path=weights_path)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 对模型进行打分，本函数按batch计算在某些输入数据上模型的误差，
# score=model.evaluate(x_test,y_test,batch_size=64)
# print(score)

# 用一张图片测试模型
# im = cv2.resize(cv2.imread('E:/ImageProce/DenseNet/DenseNet-Keras-master/resources/cat32.jpg'), (32, 32)).astype(np.float32)

# im=cv2.imread('E:/ImageProce/DenseNet/DenseNet-Keras-master/resources/cat32.jpg')
im=cv2.imread('E:/ImageProce/DenseNet/DenseNet-Keras-master/resources/airplane.jpg')
plt.imshow(im[...,-1::-1])
# 因为opencv读取进来的是bgr顺序的，而imshow需要的是rgb顺序，因此需要先反过来
plt.show()
# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)
out = model.predict(im)
print(out)

classes = []
with open('resources/cifar10_classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))

print(classes)

print('Prediction: '+str(classes[np.argmax(out)]))





