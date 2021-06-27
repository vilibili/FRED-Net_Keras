from keras.layers import Input, Add, BatchNormalization, Convolution2D,\
    Activation, LeakyReLU
from keras.models import Model
from model.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def FRED_Net(height = 256, width = 256, num_class = 2):
    kernel = 3
    pool_size = (2, 2)
    alpha = 0.8
    regularizer_l2 = None#l2 l=0.0001
    #lr = 0.01
    #weight decay = 0.0005

    img_input = Input(shape=(height, width, 3))
    x = Convolution2D(64, (kernel, kernel), padding="same")(img_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_2 = Convolution2D(128, (1,1))(pool_1)
    skip_2 = BatchNormalization()(skip_2)

    x = Add()([x, skip_2])
    x = LeakyReLU(alpha)(x)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_3 = Convolution2D(256, (1, 1))(pool_2)
    skip_3 = BatchNormalization()(skip_3)

    x = Add()([x, skip_3])
    x = LeakyReLU(alpha)(x)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(512, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_4 = Convolution2D(512, (1, 1))(pool_3)
    skip_4 = BatchNormalization()(skip_4)

    x = Add()([x, skip_4])
    x = LeakyReLU(alpha)(x)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(x)

    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([pool_4, mask_4])

    x = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(256, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_5 = Convolution2D(256, (1, 1))(unpool_1)
    skip_5 = BatchNormalization()(skip_5)

    x = Add()([x, skip_5])
    x = LeakyReLU(alpha)(x)

    unpool_2 = MaxUnpooling2D(pool_size)([x, mask_3])

    x = Convolution2D(256, (kernel, kernel), padding="same")(unpool_2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(128, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_6 = Convolution2D(128, (1, 1))(unpool_2)
    skip_6 = BatchNormalization()(skip_6)

    x = Add()([x, skip_6])
    x = LeakyReLU(alpha)(x)

    unpool_3 = MaxUnpooling2D(pool_size)([x, mask_2])

    x = Convolution2D(128, (kernel, kernel), padding="same")(unpool_3)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(64, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)

    skip_7 = Convolution2D(64, (1, 1))(unpool_3)
    skip_7 = BatchNormalization()(skip_7)

    x = Add()([x, skip_7])
    x = LeakyReLU(alpha)(x)

    unpool_4 = MaxUnpooling2D(pool_size)([x, mask_1])

    x = Convolution2D(64, (kernel, kernel), padding="same")(unpool_4)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Convolution2D(num_class, (kernel, kernel), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)

    output = Activation('softmax')(x)
    model = Model(inputs=img_input, outputs=output, name="FRED-Net")
    return model