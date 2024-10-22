#code by hector anaya 

import tensorflow as tf
import cv2
from tensorflow.keras.layers import InputLayer, LeakyReLU,Input,Lambda, Layer, Dense, Conv2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.models import clone_model
import numpy as np
from keras.initializers import RandomNormal
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import glob
from numpy import zeros
from numpy import ones
from PIL import Image
from numpy.random import randint
import os
#from keras.utils.vis_utils import plot_model
import time, gc
#gpus = tf.config.list_physical_devices('GPU')
#
#if gpus:
#  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.set_visible_devices(gpus[2], 'GPU')
#    logical_gpus = tf.config.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
#    # Visible devices must be set before GPUs have been initialized
#    print(e)
#from numpy.random import seed
#seed(1)
#tf.keras.utils.set_random_seed(2)



#test folder name
folder='Test_1/'
os.mkdir(folder)
#--Parameters
STYLE_LAYERS = ('conv2d_1', 'conv2d_3')
STYLE_LAYERS2=(2,8)
STYLE_LAYERS_SIZE = (256, 64)
STYLE_LAYERS_CHANNELS = (32, 64)
STYLE_LAYERS_MEAN = (2e-7, -2e-5)
STYLE_LAYERS_STD = (0.05, 0.03)
CONTENT_LAYER = ('block4_conv2',)
learning_rate = 0.0002 / 5
beta1 = 0.5

batch_size = 1  # Size of image batch to apply at each iteration.
max_epoch = 1000

img_channel = 3
img_size = 512
img_x = 512
img_y = 512
padding_l = 0
padding_r = 0
padding_t = 0
padding_d = 0
gt_channel = 1

style_size = 512

sample_batch = 4
z_size = 400




import csv

def log_loss_to_csv(epoch, gen_loss, disc_loss,disc_loss2,g_loss_adversarial,g_loss_tv,g_mae,g_loss_retinal,g_loss_patho,g_loss_severity, csv_filename):
    # Check if the CSV file already exists; if not, create it with the header
    write_header = not os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header if the file is newly created
        if write_header:
            writer.writerow(['Epoch', 'Generator Loss', 'Discriminator Loss 1', 'Discriminator Loss 2', 'Adversarial Loss','TV loss','MAE loss','Retinal Loss','Patho Loss','Severity loss' ])

        # Write the loss values for the current epoch
        writer.writerow([epoch, gen_loss.numpy(), disc_loss.numpy(),disc_loss2.numpy(),g_loss_adversarial.numpy(),g_loss_tv.numpy(),g_mae.numpy(),g_loss_retinal.numpy(),g_loss_patho.numpy(),g_loss_severity.numpy()])



leaky_rectify_alpha = 0.01
regular_factor_l1 = 0.
regular_factor_l2 = 5e-4  # weight_decay
def custom_activation(x):
    # Define your custom activation logic here
    return tf.keras.activations.relu(x, leaky_rectify_alpha)
def conv_params(filters, **kwargs):
    """default Conv2d arguments"""
    args = {
        'filters': filters,
        'kernel_size': (3, 3),
        'padding': 'same',
        'activation': lambda x: tf.keras.activations.relu(x, leaky_rectify_alpha),
        'use_bias': True,
        'kernel_initializer': 'zero',
        'bias_initializer': 'zero',
    }
    args.update(kwargs)
    return args


def pool_params(**kwargs):
    """default MaxPool2d/RMSPoolLayer arguments"""
    args = {
        'pool_size': (3,3),
        'strides': (2, 2),
    }
    args.update(kwargs)
    return args


def dense_params(num_units, **kwargs):
    """default dense layer arguments"""
    args = {
        'units': num_units,
        'activation': lambda x: tf.keras.activations.relu(x, leaky_rectify_alpha),
        'kernel_initializer': 'zero',
        'bias_initializer': 'zero',
    }
    args.update(kwargs)
    return args

class RMSPoolLayer(tf.keras.layers.Layer):
    """Use RMS(Root Mean Squared) as pooling function."""

    def __init__(self,pool_size=(3,3),strides=None):
        super(RMSPoolLayer, self).__init__()
        self.pool_size=pool_size
        self.strides=strides
    def build(self, input_shape):
        super(RMSPoolLayer, self).build(input_shape)

    def call(self,inputs):
        squared_inputs = tf.square(inputs)
        output = tf.nn.avg_pool2d(squared_inputs, ksize =self.pool_size, strides=self.strides, padding='VALID')
        return tf.sqrt(output + tf.keras.backend.epsilon())
def cp(filters, **kwargs):
    args = {
        'filters': filters,
        'kernel_size': (4, 4),
    }
    args.update(kwargs)
    return conv_params(**args)
cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
}
n = 32

layers = [
    (InputLayer, {'input_shape': (cnf['h'], cnf['w'], 3)}),
    (Conv2D, cp(n, strides=(2, 2))),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(n,
                kernel_regularizer=l1_l2(regular_factor_l1, regular_factor_l2),
                bias_regularizer=l1_l2(regular_factor_l1, regular_factor_l2))),
    (MaxPool2D, pool_params()),
    (Conv2D, cp(2 * n, strides=(2, 2))),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(2 * n)),
    (Conv2D, cp(2 * n)),
    (MaxPool2D, pool_params()),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(4 * n)),
    (Conv2D, cp(4 * n)),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(4 * n)),
    (MaxPool2D, pool_params()),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(8 * n)),
    (Conv2D, cp(8 * n)),
    (ZeroPadding2D, {'padding': 2}), (Conv2D, cp(8 * n)),
    (MaxPool2D, pool_params()),
    (Conv2D, cp(16 * n)),
    (RMSPoolLayer, pool_params()),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}),
    (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), 
    (MaxPooling1D, {'pool_size': 2}),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}),
    (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), 
    (MaxPooling1D, {'pool_size': 2}),
    (Flatten, {}),
    (Dense, dense_params(1, activation='linear')),
]




def get_detector():
    """return the detector singleton model, which is used to build the layers model"""
    global _detector
    if _detector is not None:
        return _detector
    with tf.name_scope('detector_prototype'):
        _detector = Sequential(name=cnf['name'])
        i=0
        for layer, kwargs in layers:
            if i<2:
                print(layer(**kwargs))
                i=i+1 
            if layer is Dense and i==2:
                print(layer(**kwargs))
                i=3
            if 'activation' not in kwargs:
                _detector.add(layer(**kwargs))
                if layer is InputLayer:
                    _detector.add(Lambda(lambda x: x, name='my_input'))
            else:
                del kwargs['activation']
                new_layer = layer(**kwargs)
                new_layer.related_activation = LeakyReLU(leaky_rectify_alpha, name=new_layer.name+'_act')
                _detector.add(new_layer)
                _detector.add(new_layer.related_activation)
    return _detector




def get_style_model(image, mask, with_feature_mask_from=None):
    if mask is not None:
        image = (image + 1) * ((mask + 1) / 2) - 1
    if image.shape[1] != 448:
        image =cv2.resize(image[0], (448, 448))

    model = get_layers_model(image, STYLE_LAYERS + ('dense_3',), 'style_model',
                                      with_projection_output_from=with_feature_mask_from)
    return model




def gauss_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.max(kernel)





vgg = tf.keras.applications.VGG16(
        weights='imagenet',
        input_shape=(512, 512, 3),
        include_top=False
    )

intermediate_model = tf.keras.models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in CONTENT_LAYER])




_detector = None
models = {}

proto = get_detector()
input_layer = Input(shape=((448,448,3)))
model = clone_model(proto, input_layer)
model.load_weights('data/detector.h5', by_name=True)

def get_content_features(image, mask):
    image = (image + 1) * 127.5
    if mask is not None:
        image = image * ((mask + 1) / 2)

    img_features = {}

    if image.shape[1] != 512:
        image = tf.image.resize(image, [512, 512])


    img_pre = tf.keras.applications.vgg16.preprocess_input(image)

    net = intermediate_model(img_pre)

    for i, layer in enumerate(CONTENT_LAYER):
        img_features[layer] = net[i]

    return img_features


def get_retinal_loss(img, syn, mask):

    img_features = get_content_features(img, mask)
    syn_features = get_content_features(syn, mask)

    content_lossE = 0
    for content_layer in CONTENT_LAYER:
        coff = float(1.0 / len(CONTENT_LAYER))
        img_content = img_features[content_layer]
        syn_content = syn_features[content_layer]
        content_lossE += coff * tf.reduce_mean(tf.abs(img_content - syn_content))

    content_loss = tf.reduce_mean(content_lossE)

    return content_loss

def get_tv_loss(img, mask, input_mask=None):


    img = img*((mask+1)/2)

    if input_mask is not None:
        x = tf.reduce_sum(input_mask[:, :-1, :, :] * tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :])) / (1e-8 + 3*tf.reduce_sum(input_mask))
        y = tf.reduce_sum(input_mask[:, :, :-1, :] * tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :])) / (1e-8 + 3*tf.reduce_sum(input_mask))
    else:
        x = tf.reduce_mean(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
        y = tf.reduce_mean(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    return x+y
    

def process_image(img, mask,img_detector):
    img=cv2.resize(img[0],(448,448))
    img=tf.expand_dims(img, axis=0)
    act_input={}
    for index,layer_name,model1,model2, size, mean, std in zip(STYLE_LAYERS2,STYLE_LAYERS,STYLE_MODELS,STYLE_MODELS_2,STYLE_LAYERS_SIZE,STYLE_LAYERS_MEAN,STYLE_LAYERS_STD):
        key=size
        x=model1(img)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            output_grads=model2(x)
        output = tape.gradient(output_grads,x)
        value=tf.image.resize((output - mean)/std, [size,size])
        value=value.numpy()
        act_input[key]=value
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img)
        output=aux_model_patho(img)
    output = tape.gradient(output,img)
    


    projection = (output - 1e-6) / 0.05
    projection = tf.clip_by_value(projection, -0.5, 0.5)
    projection = projection * 2
    projection = tf.abs(projection)
    projection = tf.reduce_mean(projection, 3, keepdims=True)
    projection=cv2.convertScaleAbs(projection.numpy()[0],alpha=0.5,beta=0.1)
    projection=cv2.dilate(projection, np.ones((5,5), np.uint8) , iterations=1)
    projection=projection.astype(np.float32)
    projection=tf.expand_dims(projection,axis=-1)

    projection = tf.nn.conv2d(tf.expand_dims(projection,axis=0), gauss_kernel(31, 10)[..., None, None], [1,1,1,1],'SAME')  # gauss blur
    projection = projection / (tf.reduce_max(projection) + 1e-7)
    projection=projection.numpy()


    # Define a Python function for thresholding
    def threshold_image(gray):
        # Scale the values from [0, 1] to [0, 255] and convert to uint8
        scaled_image = gray[0]* 255
        scaled_image=scaled_image.astype(np.uint8)
        # Apply Otsu's thresholding to create a binary mask
        _, thresholded = cv2.threshold(scaled_image, 0, 255,cv2.THRESH_OTSU)
        # Expand dimensions to match the expected output shape
        thresholded = thresholded[None, ..., None]

        return thresholded

    binary_mask =threshold_image(projection)
    
    binary_mask256 = tf.cast(tf.image.resize(binary_mask, [256, 256], method=tf.image.ResizeMethod.BILINEAR) > 0, tf.float32)
    binary_mask64 = tf.cast(tf.image.resize(binary_mask, [64, 64], method=tf.image.ResizeMethod.BILINEAR) > 0, tf.float32)
    binary_mask256=binary_mask256.numpy()
    binary_mask64=binary_mask64.numpy()
    masked_act_input = {
        256: binary_mask256 * act_input[256],
        64: binary_mask64 * act_input[64],
    }

    
    return act_input,masked_act_input



def resizeandrescale(img):
    img=img.astype(np.float32)
    img=img/255
    img= (img - 0.5) * 2.0
    img=img[...,::-1]
    return img
  



def generate_real_samples(dataset, n_samples, patch_shape):
# unpack dataset
    trainA, trainB,trainC = dataset
    
    # choose random instances
    ix = randint(0, len(trainA), n_samples)
    # retrieve selected images
    X1, X2, X3 = trainA[ix[0]], trainB[ix[0]], trainC[ix[0]]
    X1=resizeandrescale(cv2.imread(X1))
    X2=resizeandrescale(cv2.imread(X2))
    X3=resizeandrescale(cv2.imread(X3))
    X1=tf.expand_dims(X1,axis=0)
    X2=tf.expand_dims(X2,axis=0)
    y = ones((n_samples,patch_shape,patch_shape,  1))
    return [X1, X2, X3], y


def generate_fake_samples(g_model, img,act256,act64,z, patch_shape):
# generate fake instance
    X = g_model([img,act256,act64,z],training=True)
    y = zeros((len(X),patch_shape,patch_shape, 1))
    return X, y




aux_model_patho=Model(inputs=model.input,outputs=model.get_layer(index=44).output)




def patho_loss(img):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img)
        output=aux_model_patho(img)
    output = tape.gradient(output,img)
    return output
def summarize_performance(step, g_model,d_model, dataset, n_samples=1):
# select a sample of input images
    [X_realA,X_realB,mask],_=generate_real_samples(dataset,n_samples,n_patch) #generate batch of real images and label
    # generate a batch of fake samples
    
    z_sample = np.random.normal(0, 0.001, size=[1,400]).astype(np.float32)
    act,act_mask=process_image(X_realA.numpy(),mask,model)
    X_fakeB,_ = generate_fake_samples(g_model,X_realB,act_mask[256],act_mask[64],z_sample,n_patch)
    syn = ((X_fakeB+ 1) / 2.0)*((mask+1)/2)
    syn=syn.numpy()
    cv2.imwrite(folder+'/'+str(step)+'.png',  cv2.cvtColor(syn[0]*255, cv2.COLOR_BGR2RGB))

    # scale all pixels from [-1,1] to [0,1]
    X_realA = ((X_realA + 1) / 2.0)*mask
    X_realB = ((X_realB+ 1) / 2.0)*mask 
    X_fakeB = ((X_fakeB+ 1) / 2.0)*mask 
# plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
        # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i])
        # save plot to file
    filename1 =folder+'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = folder+f'generator_server_{folder[:-1]}_{(step+1)/337}.h5'
    filename3=folder+'discriminator.h5'
    d_model.save(filename3)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

resize_and_rescale = tf.keras.Sequential([
  RandomRotation((-0.1, 0.1),fill_mode='constant',fill_value=-1)
])





aux_model_conv2d_1=Model(inputs=model.input,outputs=model.get_layer(STYLE_LAYERS[0]).output) #model conv2d_1
aux_model_conv2d_1_out=Model(inputs=model.get_layer(index=STYLE_LAYERS2[0]).input,outputs=model.get_layer(index=44).output) #model conv2d_1




aux_model_conv2d_3=Model(inputs=model.input,outputs=model.get_layer(STYLE_LAYERS[1]).output) #model conv2d_1
aux_model_conv2d_3_out=Model(inputs=model.get_layer(index=STYLE_LAYERS2[1]).input,outputs=model.get_layer(index=44).output) #model conv2d_1




STYLE_MODELS=(aux_model_conv2d_1,aux_model_conv2d_3)
STYLE_MODELS_2=(aux_model_conv2d_1_out,aux_model_conv2d_3_out)




aux_model_dense3=Model(inputs=model.input,outputs=model.get_layer('dense_3').output)




aux_model_dense3.summary()



#hiper parameters
#adversarial
w_adv = 1
#pathological
w_patho = 1e3 # 50e6
#retinal_details
w_retinal = 1
#tv
w_tv = 100
#severity
w_severity = 10



#dataset load, only train subset


data_img=glob.glob('data/IDRiD/train_512/*.jpg')
data_mask='data/IDRiD/mask.png'
data_gt=glob.glob('data/IDRiD/train_512/*_VS.png')


#verify same size
print(len(data_img),len(data_mask),len(data_gt))



#they need to have the same name, so when sorted, they are with its corresponding mask and groundtruth
data_img.sort()
data_gt.sort()
data_mask.sort()


dataset=[data_img,data_gt,data_mask]


z_sample = np.random.normal(0, 0.001, size=[1,400]).astype(np.float32)



# define a encoder block

def define_encoder_block(layer_in, n_filters, batchnorm=True):

    init =tf.random_normal_initializer(0., 0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same',
         kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
    
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters,n, dropout=True):
    # weight initialization
    init =tf.random_normal_initializer(0., 0.02)
# add upsampling layer
    g=tf.image.resize(layer_in,(2**(n+1),2**(n+1)),method='nearest')
    g=Conv2D(n_filters,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(g)
# add batch normalization
    g = BatchNormalization()(g, training=True)
# conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g





def define_discriminator(image_shape):
        # weight initialization
    init =tf.random_normal_initializer(0., 0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    
    # define model
    model = Model([in_src_image, in_target_image], d)
    return model




discriminator=define_discriminator((512,512,3))


discriminator.summary()



def define_generator(act_input256,act_input64,z_shape,image_shape=(256,256,3)):
# weight initialization
    init =tf.random_normal_initializer(0., 0.02)
    # image input
    act256=Input(shape=act_input256)
    act64=Input(shape=act_input64)
    z=Input(shape=z_shape)
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    merge=Concatenate()([e1,act256])
    e2 = define_encoder_block(merge, 128)
    e3 = define_encoder_block(e2, 256)

    merge2=Concatenate()([e3,act64])

    e4 = define_encoder_block(merge2, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    zdense = Dense(4*4*64, kernel_initializer=init)(z)
    zl=LeakyReLU(alpha=0.2)(zdense)
    zb=tf.reshape(zl,[-1,4,4,64])
    
    # decoder model: CD512-CD512-CD512-C512-C256-128-64
    d1=Concatenate()([zb,e7])
    d2 = decoder_block(d1, e6, 512,n=2)
    d3 = decoder_block(d2, e5, 512,n=3)
    d4 = decoder_block(d3, e4, 512,n=4)
    d5 = decoder_block(d4, e3, 256, dropout=False,n=5)
    d6 = decoder_block(d5, e2, 128, dropout=False,n=6)
    d7 = decoder_block(d6, e1, 64, dropout=False,n=7)
    # output
    g=tf.image.resize(d7,(2**(8+1),2**(8+1)),method='lanczos5')
    g=Conv2D(3,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model([in_image,act256,act64,z], out_image)
    return model

generator=define_generator(act_input256=(256,256,32),act_input64=(64,64,64),z_shape= (400,),image_shape=(512,512,3))


d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=beta1)
g_optimizer =  tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=beta1)


##if in a server use:
# import matplotlib
# matplotlib.use('Agg') 

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
loss_mae=tf.keras.losses.MeanAbsoluteError()
patho_loss_fun=tf.keras.losses.MeanSquaredError()
sev_loss_fun=tf.keras.losses.MeanSquaredError()


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss=-tf.reduce_mean(disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (0 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

#define the gradient penalty function
def gradient_penalty(x, x_gen,gtt):
    x = tf.cast(x, tf.float32)
    x_gen = tf.cast(x_gen, tf.float32)
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    diff=x_gen-x
    x_hat = x+(epsilon*diff)
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator([gtt,x_hat],training=True)
    gradients = t.gradient(d_hat, x_hat)

    ddx = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
    return d_regularizer
#define the wasserstein loss for discriminator 
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss=-tf.reduce_mean(disc_real_output)
    generated_loss=tf.reduce_mean(disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss




@tf.function
def train_step_3(target,input_image,gen_output,mask,act256,act64,n_batch,g_patho,g_sev,tar,inp,mas,ac):
    for i in range(5):
        z= tf.random.normal(shape=(batch_size, 400))  #sample
        with tf.GradientTape() as disc_tape:
            gen_output_2 = (generator([inp[i],ac[i][256],ac[i][64],z],training=True)+1)*((mask+1)/2)-1
            disc_real_output = discriminator([inp[i], tar[i]], training=True)
            disc_generated_output = discriminator([inp[i], gen_output_2], training=True)
            penalty=gradient_penalty(tar[i],gen_output_2,inp[i])
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)+penalty*10
        discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))

    z= tf.random.normal(shape=(batch_size, 400))  #sample
    with tf.GradientTape() as gen_tape:
        gen_output = (generator([input_image,act256,act64,z],training=True)+1)*((mask+1)/2)-1
            
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        g_ret=get_retinal_loss(target,gen_output,mask)

        g_loss_tv=get_tv_loss(gen_output,mask)
        gen_total_loss_2, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        gen_total_loss=gen_total_loss_2+ w_patho * g_patho+ w_severity * g_sev  + w_retinal * g_ret  + w_tv * g_loss_tv
        



    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)


    g_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))

    return disc_loss,gen_total_loss_2,gen_total_loss,gen_gan_loss,g_loss_tv,gen_l1_loss,g_ret#,g_ret





epochs=2000

n_patch = discriminator.output_shape[1] 
n_batch=1
i=0
bat_per_epo = int(len(data_img) / n_batch)
# calculate the number of training iterations
n_steps = bat_per_epo * epochs
for epoch in range(epochs):
    print("\nStart epoch", epoch)
    for step in range(bat_per_epo):
        i=i+1
        # Train the discriminator & generator on one batch of real images.
        [X_realA,X_realB,mask],labelsReal=generate_real_samples(dataset,n_batch,n_patch) #generate batch of real images and label
        all_images=tf.concat([X_realA,X_realB,tf.expand_dims(mask,axis=0)],axis=3)
        resized=resize_and_rescale(all_images)
        target,gt,mask=tf.split(resized, num_or_size_splits=3, axis=3)
        _,act_mask=process_image(target.numpy(),mask,model)
        z_sample = np.random.normal(0, 0.001, size=[1,400]).astype(np.float32)
        X_fakeB, labelsFake = generate_fake_samples(generator,gt,act_mask[256],act_mask[64],z_sample,n_patch)
        xx=[]
        gg=[]
        mm=[]
        aa=[]
        for w in range(5):
            [A,B,m],labelsReal=generate_real_samples(dataset,n_batch,n_patch) #generate batch of real images and label
            all=tf.concat([A,B,tf.expand_dims(m,axis=0)],axis=3)
            res=resize_and_rescale(all)
            t,g,ma=tf.split(res, num_or_size_splits=3, axis=3)
            _,a_m=process_image(t.numpy(),ma,model)
            xx.append(t)
            gg.append(g)
            mm.append(ma)
            aa.append(a_m)
        img_det_myinput=patho_loss(tf.image.resize((target+1)*((mask+1)/2)-1,(448,448)))
        syn_det_myinput=patho_loss(tf.image.resize((X_fakeB+1)*((mask+1)/2)-1,(448,448)))
        img_det_dense=aux_model_dense3(tf.image.resize((target+1)*((mask+1)/2)-1,(448,448)))
        syn_det_dense=aux_model_dense3(tf.image.resize((X_fakeB+1)*((mask+1)/2)-1,(448,448)))
        g_loss_patho=patho_loss_fun(img_det_myinput,syn_det_myinput)   
        g_loss_severity=sev_loss_fun(img_det_dense,syn_det_dense)
        # Logging.
        d_loss1,d_loss2,g_loss,g_loss_adversarial,g_loss_tv,g_mae,g_loss_retinal=train_step_3(target,gt,X_fakeB,mask,act_mask[256],act_mask[64],1,g_loss_patho,g_loss_severity,xx,gg,mm,aa)
        log_loss_to_csv(i, g_loss, d_loss1,d_loss2,g_loss_adversarial,g_loss_tv,g_mae,g_loss_retinal,g_loss_patho,g_loss_severity, folder+f'loss_log_server_{folder[:-1]}.csv')
        if step % 5 == 0:
        # Print metrics
            print("discriminator loss at step %d: %.2f" % (step, d_loss1))
            print("discriminator loss at step %d: %.2f" % (step, d_loss2))
            print("adversarial loss at step %d: %.2f" % (step, g_loss))
            print(f" patho : {g_loss_patho}, gan : {g_loss_adversarial}, severity : {g_loss_severity}, tv: {g_loss_tv}, retinal: {g_loss_retinal},mae {g_mae}")
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, generator,discriminator, dataset)