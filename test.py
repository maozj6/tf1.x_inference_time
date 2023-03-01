import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import  Dropout, Flatten

from tensorflow.keras.models import Model
import numpy as np
import time

ONE_BYTE_SCALE = 1.0 / 255.0
ONE_BYTE_SCALE = 1.0 / 255.0

if __name__ == '__main__':
    def conv2d(filters, kernel, strides, layer_num, activation='relu'):
        """
        Helper function to create a standard valid-padded convolutional layer
        with square kernel and strides and unified naming convention

        :param filters:     channel dimension of the layer
        :param kernel:      creates (kernel, kernel) kernel matrix dimension
        :param strides:     creates (strides, strides) stride
        :param layer_num:   used in labelling the layer
        :param activation:  activation, defaults to relu
        :return:            tf.keras Convolution2D layer
        """
        return Convolution2D(filters=filters,
                             kernel_size=(kernel, kernel),
                             strides=(strides, strides),
                             activation=activation,
                             name='conv2d_' + str(layer_num))


    def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
        drop = 0.2
        img_in = Input(shape=input_shape, name='img_in')
        x = core_cnn_layers(img_in, drop)
        x = Dense(100, activation='relu', name='dense_1')(x)
        x = Dropout(drop)(x)
        x = Dense(50, activation='relu', name='dense_2')(x)
        x = Dropout(drop)(x)

        outputs = []
        for i in range(num_outputs):
            outputs.append(
                Dense(1, activation='linear', name='n_outputs' + str(i))(x))

        model = Model(inputs=[img_in], outputs=outputs, name='linear')
        return model


    def core_cnn_layers(img_in, drop, l4_stride=1):
        """
        Returns the core CNN layers that are shared among the different models,
        like linear, imu, behavioural

        :param img_in:          input layer of network
        :param drop:            dropout rate
        :param l4_stride:       4-th layer stride, default 1
        :return:                stack of CNN layers
        """
        x = img_in
        x = conv2d(24, 5, 2, 1)(x)
        x = Dropout(drop)(x)
        x = conv2d(32, 5, 2, 2)(x)
        x = Dropout(drop)(x)
        x = conv2d(64, 5, 2, 3)(x)
        x = Dropout(drop)(x)
        x = conv2d(64, 3, l4_stride, 4)(x)
        x = Dropout(drop)(x)
        x = conv2d(64, 3, 1, 5)(x)
        x = Dropout(drop)(x)
        x = Flatten(name='flattened')(x)
        return x




    model=default_n_linear(1)
    test=np.random.rand(200,1,120, 160, 3)
    test = test.astype(np.float32)  #


    def count():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


    print(count())

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for var in tf.global_variables():
        shape = var.shape
        array = np.asarray([dim.value for dim in shape])
        mulValue = np.prod(array)

        Total_params += mulValue
        if var.trainable:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')






    time1 = time.time()
    for i in range(len(test)):
        out = model(test[i])
    print("warm-up time is "+str(time.time()-time1))
    test2=np.random.rand(200,1,120, 160, 3)
    test2 = test2.astype(np.float32)
    time2 = time.time()
    for i in range(len(test)):
        out = model(test[i])
    total=time.time()-time2
    print("warm-up time is "+str(total))
    print("average time is "+str(total/200)+" per inference")
    print("end")
