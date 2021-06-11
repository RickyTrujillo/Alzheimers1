import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, concatenate,AveragePooling2D, Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD


k_init = keras.initializers.glorot_uniform()
b_init = keras.initializers.constant(value=0.2)
input_shape = Input(shape=(256, 256, 1))


def inception_module(input_x, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool,
                     name=None):
    path1 = Conv2D(filter_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)

    path2 = Conv2D(filter_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)
    path2 = Conv2D(filter_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path2)

    path3 = Conv2D(filter_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(input_x)
    path3 = Conv2D(filter_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path3)

    path4 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(input_x)
    path4 = Conv2D(filter_pool, (1, 1), padding='same', activation='relu', kernel_initializer=k_init,
                   bias_initializer=b_init)(path4)

    output = concatenate([path1, path2, path3, path4], axis=3, name=None)
    return output


def auxiliary_module(x, name=None):
    x_aux = AveragePooling2D((5, 5), strides=(3, 3))(x)
    x_aux = Conv2D(128, (1, 1), padding='same', activation='relu')(x_aux)
    x_aux = Flatten()(x_aux)
    x_aux = Dense(1024, activation='relu')(x_aux)
    x_aux = Dropout(0.7)(x_aux)
    x_aux = Dense(4, activation='softmax')(x_aux)
    return x_aux


def model_architecture(test_images, test_labels, valid_images, valid_labels, train_images, train_labels):
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=k_init,
               bias_initializer=b_init, name='conv1_7x7/2')(input_shape)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3/2')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2b_3x3/1')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3/2')(x)

    x = inception_module(x, filter_1x1=64, filter_3x3_reduce=96, filter_3x3=126, filter_5x5_reduce=16, filter_5x5=32,
                         filter_pool=32, name='inception_3a')

    x = inception_module(x, filter_1x1=128, filter_3x3_reduce=128, filter_3x3=192, filter_5x5_reduce=32, filter_5x5=96,
                         filter_pool=64,
                         name='inception_3b')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3/2')(x)

    x = inception_module(x, filter_1x1=192, filter_3x3_reduce=96, filter_3x3=208, filter_5x5_reduce=16, filter_5x5=48,
                         filter_pool=64,
                         name='inception_4a')

    #x1 = auxiliary_module(x, name='auxiliary_1')

    x = inception_module(x, filter_1x1=160, filter_3x3_reduce=112, filter_3x3=224, filter_5x5_reduce=24, filter_5x5=64,
                         filter_pool=64,
                         name='inception_4b')

    x = inception_module(x, filter_1x1=128, filter_3x3_reduce=128, filter_3x3=256, filter_5x5_reduce=24, filter_5x5=64,
                         filter_pool=64,
                         name='inception_4c')

    x = inception_module(x, filter_1x1=112, filter_3x3_reduce=144, filter_3x3=288, filter_5x5_reduce=32, filter_5x5=64,
                         filter_pool=64,
                         name='inception_4d')

    #x2 = auxiliary_module(x, name='auxiliary_2')

    x = inception_module(x, filter_1x1=256, filter_3x3_reduce=160, filter_3x3=320, filter_5x5_reduce=32, filter_5x5=128,
                         filter_pool=128,
                         name='inception_4e')

    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3/2')(x)

    x = inception_module(x, filter_1x1=256, filter_3x3_reduce=160, filter_3x3=320, filter_5x5_reduce=32, filter_5x5=128,
                         filter_pool=128,
                         name='inception_5a')

    x = inception_module(x, filter_1x1=384, filter_3x3_reduce=192, filter_3x3=384, filter_5x5_reduce=48, filter_5x5=128,
                         filter_pool=128,
                         name='inception_5b')

    x = GlobalAveragePooling2D(name='gblavgpool5_3x3/1')(x)
    x = Dropout(0.4)(x)
    x = Dense(4, activation='softmax', name='output')(x)
    # model = Model(input_shape, [x, x1, x2], name='inception_v1')
    model = Model(input_shape, x, name='inception_v1')
    model.summary()

    sgd = SGD(learning_rate=1e-4, momentum=0.4, decay=0.01, nesterov=False)
    # model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  #loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', loss_weights=0.3, optimizer=sgd, metrics=['accuracy'])
    # history = model.fit(train_images, [train_labels, train_labels, train_labels], validation_data=(valid_images, [valid_labels, valid_labels, valid_labels]), epochs=5, batch_size=256)
    print("Evaluation on Training and Validation Data")
    history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=10, batch_size=32)
    print("Evaluation on Testing Data")
    test_history = model.evaluate(test_images, test_labels, batch_size=256)