import numpy as np
from tensorflow.python.keras.engine import get_source_inputs
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, \
    Dense, Conv2D, Convolution2D, Conv2DTranspose, UpSampling2D, \
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    BatchNormalization, Activation, Dropout, \
    Concatenate, concatenate, merge, \
    Reshape, Flatten, ZeroPadding2D
from tensorflow.python.keras.regularizers import l2
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import backend as K


def unet_half(in_shape, num_classes, activation='softmax'):
    # TODO: Make sure I give credit to
    #  https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/
    #  blob/master/unet-segmentation.ipynb for code.
    def down_block(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(x)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(conv)
        conv = Activation('relu')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    num_filters = np.array([32, 64, 128, 256, 512])
    input_layer = Input(in_shape)
    p0 = input_layer
    c1, p1 = down_block(p0, num_filters[0])
    c2, p2 = down_block(p1, num_filters[1])
    c3, p3 = down_block(p2, num_filters[2])
    c4, p4 = down_block(p3, num_filters[3])

    f1 = Flatten()(p4)
    output = Dense(units=num_classes)(f1)
    output = Activation(activation)(output)
    model = Model(inputs=input_layer, outputs=output, name='unet-half')
    return model


def unet(in_shape, num_classes):
    # TODO: Make sure I give credit to
    #  https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/
    #  blob/master/unet-segmentation.ipynb for code.
    def down_block(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(x)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(conv)
        conv = Activation('relu')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def up_block(x, skip, filters, kernel_size=(3, 3),
                 strides=1, padding='same'):
        u1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                             strides=2, padding=padding)(x)
        cat = merge.concatenate([u1, skip])
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(cat)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(conv)
        conv = Activation('relu')(conv)
        return conv

    def bottleneck(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(x)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(conv)
        conv = Activation('relu')(conv)
        return conv

    num_filters = [32, 64, 128, 256, 512]
    input = Input(in_shape)
    p0 = input
    c1, p1 = down_block(p0, num_filters[0])
    c2, p2 = down_block(p1, num_filters[1])
    c3, p3 = down_block(p2, num_filters[2])
    c4, p4 = down_block(p3, num_filters[3])

    bn = bottleneck(p4, num_filters[4])

    u1 = up_block(bn, c4, num_filters[3])
    u2 = up_block(u1, c3, num_filters[2])
    u3 = up_block(u2, c2, num_filters[1])
    u4 = up_block(u3, c1, num_filters[0])

    output = Conv2D(filters=1, kernel_size=(1, 1))(u4)
    output = Flatten()(output)
    output = Dense(units=num_classes)(output)
    output = Activation('softmax')(output)
    model = Model(inputs=input, outputs=output, name='unet')
    return model


def wang(in_shape, num_classes, activation='softmax'):
    def down_block(x, filters, kernel_size=(5, 5), strides=1, padding='valid'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding)(x)
        conv = Activation('relu')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    num_filters = np.array([64, 128])
    input_layer = Input(in_shape)
    p0 = input_layer
    c1, p1 = down_block(p0, num_filters[0])
    c2, p2 = down_block(p1, num_filters[1])

    f1 = Flatten()(p2)
    output = Dense(units=num_classes)(f1)
    output = Activation(activation)(output)
    model = Model(inputs=input_layer, outputs=output, name='wang')
    return model


# https://github.com/titu1994/DenseNet/blob/master/densenet.py
def dense_net(input_shape=None, classes=1000, bottleneck=True, reduction=0.5,
              dropout_rate=0.0, weight_decay=1e-4, include_top=True,
              weights=None, fuse=False, input_tensor=None,
              activation='softmax'):
    """
    Args:
        fuse: BatchNormalization default fuse=True combines individual
            operations into a single kernel for faster performance.
            Set fuse=False if dtype=float64 since it is not a supported dtype.
    """

    def setup(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12,
              nb_filter=-1, nb_layers_per_block=-1, bottleneck=False,
              reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
              subsample_initial_block=False, include_top=True, weights=None,
              input_tensor=None, classes=10, activation='softmax'):
        """
        Instantiate the DenseNet architecture.
        # Arguments
            input_shape: input shape tuple, only to be specified if
                `include_top` is False (otherwise the input shape has to be
                `(32, 32, 3)` (with `channels_last` dim ordering) or
                `(3, 32, 32)` (with `channels_first` dim ordering). It should
                 have exactly 3 inputs channels, and width and height should be
                 no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial number
                of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial
                convolution and add a MaxPool2D before the dense blocks are
                added.
            include_top: whether to include the fully-connected layer at
                the top of the network.
            weights: one of `None` (random initialization) or 'imagenet'
                (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of
                `layers.Input()`) to use as image input for the model.
            classes: optional number of classes to classify images into, only
                to be specified if `include_top` is True, and if no `weights`
                argument is specified.
            activation: Type of activation at the top layer. Can be one of
                'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns: A Keras model instance.
        """

        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `cifar10` '
                             '(pre-training on CIFAR-10).')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as ImageNet with '
                             '`include_top` as true, `classes` should be 1000')

        if activation not in ['softmax', 'sigmoid', 'linear']:
            raise ValueError('activation must be one of "softmax", "sigmoid", '
                             'or "linear"')

        if activation == 'sigmoid' and classes != 1:
            raise ValueError('sigmoid activation can only be used when '
                             'classes = 1')

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=32,
                                          min_size=8,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = __create_dense_net(
            classes, img_input, include_top, depth, nb_dense_block,
            growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
            dropout_rate, weight_decay, subsample_initial_block, activation
        )

        # Ensure that the model takes into account any potential predecessors
        # of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='densenet')
        return model

    def __conv_block(ip, nb_filter, bottleneck=False,
                     dropout_rate=None, weight_decay=1e-4):
        """
        Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block
        and dropout.
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: Keras tensor with batch_norm, relu and convolution2d added
        (optional bottleneck).
        """
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, fused=fuse)(ip)
        x = Activation('relu')(x)

        if bottleneck:
            # Obtained from:
            # https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
            inter_channel = nb_filter * 4

            x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                       padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   fused=fuse)(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal',
                   padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                      dropout_rate=None, weight_decay=1e-4,
                      grow_nb_filters=True, return_concat_list=False):
        """
        Build a dense_block where the output of each conv_block is fed
        to subsequent ones.
        Args:
            x: keras tensor
            nb_layers: number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with
                the actual output
        Returns: Keras tensor with nb_layers of conv_block appended.
        """
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck,
                              dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter

    def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
        """
        Apply BatchNorm, Relu 1x1, Conv2D, optional compression,
        dropout and Maxpooling2D.
        Args:
            ip: keras tensor
            nb_filter: number of filters
            compression: calculated as 1 - reduction.
                Reduces the number of feature maps in the transition block.
            weight_decay: weight decay factor
        Returns: keras tensor, after applying batch_norm, relu-conv,
        dropout, maxpool.
        """
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, fused=fuse)(ip)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_filter * compression), (1, 1),
                   kernel_initializer='he_normal', padding='same',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x

    def __create_dense_net(nb_classes, img_input, include_top, depth=40,
                           nb_dense_block=3, growth_rate=12, nb_filter=-1,
                           nb_layers_per_block=-1, bottleneck=False,
                           reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                           subsample_initial_block=False, activation='softmax'):
        """
        Build the DenseNet model
        Args:
            nb_classes: number of classes
            img_input: tuple of shape (channels, rows, columns) or
                (rows, columns, channels)
            include_top: flag to include the final Dense layer
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Default -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                    Can be a -1, positive integer or a list.
                    If -1, calculates nb_layer_per_block from the depth of
                    the network.
                    If positive integer, a set number of layers per dense block.
                    If list, nb_layer is used as provided. Note that list size
                    must be (nb_dense_block + 1)
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks.
                Note: reduction value is inverted to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial
                convolution and add a MaxPool2D before the dense blocks are
                added.
            activation: Type of activation at the top layer. Can be one of
                'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        Returns: keras tensor with nb_layers of conv_block appended
        """

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if reduction != 0.0:
            assert 0.0 < reduction <= 1.0, \
                'reduction value must lie between 0.0 and 1.0'

        # layers in each dense block
        if type(nb_layers_per_block) is list or \
                type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == nb_dense_block, \
                'If list, nb_layer is used as provided. ' \
                'Note that list size must be (nb_dense_block)'
            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, \
                    'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (3, 3)
            initial_strides = (1, 1)

        x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal',
                   padding='same', strides=initial_strides, use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   fused=fuse)(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(
                x, nb_layers[block_idx], nb_filter, growth_rate,
                bottleneck=bottleneck, dropout_rate=dropout_rate,
                weight_decay=weight_decay
            )
            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression,
                                   weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x, nb_filter = __dense_block(
            x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
            dropout_rate=dropout_rate, weight_decay=weight_decay
        )

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, fused=fuse)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        if include_top:
            x = Dense(nb_classes, activation=activation)(x)

        return x

    return setup(input_shape, depth=121, nb_dense_block=4, growth_rate=32,
                 nb_filter=64, nb_layers_per_block=[6, 12, 24, 16],
                 bottleneck=bottleneck, reduction=reduction,
                 dropout_rate=dropout_rate, weight_decay=weight_decay,
                 subsample_initial_block=False, include_top=include_top,
                 weights=weights, input_tensor=input_tensor,
                 classes=classes, activation=activation)
