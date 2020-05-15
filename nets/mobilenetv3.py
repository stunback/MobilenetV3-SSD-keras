from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from keras import backend as K


class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init
        
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

class MobileNetV3_Large(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Large.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        net = {}  #SSD结构，net字典

        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        net['conv4_3'] = x
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        
        if self.include_top:
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')
        net['fc7'] = x

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        
        x = Conv2D(256, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (1, 1), padding='same')(x)
        net['conv6_2'] = x
        
        x = Conv2D(128, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (1, 1), padding='same')(x)
        net['conv7_2'] = x
        
        x = Conv2D(128, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (1, 1), padding='same')(x)
        net['conv8_2'] = x
        
        x = Conv2D(64, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (1, 1), padding='same')(x)
        net['conv9_2'] = x

        if plot:
            plot_model(model, to_file='./img/MobileNetv3_large.png', show_shapes=True)
        if self.include_top:
            return model
        
        return net,inputs

class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, n_class, input_tensor, alpha=1.0, include_top=True, backbone=False):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Small, self).__init__(shape, n_class, alpha)
        self.include_top = include_top
        self.backbone = backbone
        self.input_tensor = input_tensor

    def build(self, plot=False):
        """build MobileNetV3 Small.
        # Arguments
            plot: Boolean, weather to plot model.
        # Returns
            model: Model, model.
        """
        net = {} #SSD结构，net字典
        if not self.backbone:
            inputs = Input(shape=self.shape)
        else:
            inputs = self.input_tensor

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
        
        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        net['conv4_3'] = x
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        
        if self.include_top:
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, 576))(x)
        
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')
        net['fc7'] = x
        
        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        
        x = Conv2D(256, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (1, 1), padding='same')(x)
        net['conv6_2'] = x
        
        x = Conv2D(128, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (1, 1), padding='same')(x)
        net['conv7_2'] = x
        
        x = Conv2D(128, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (1, 1), padding='same')(x)
        net['conv8_2'] = x
        
        x = Conv2D(64, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (1, 1), padding='same')(x)
        net['conv9_2'] = x
        
        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)
        if self.include_top:
            return model
        
        return net