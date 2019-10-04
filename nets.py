import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils

MIN_DISP = 0.01
DISP_SCALING = 10

def pos_exp_net(target,sources,num_sources,explain=True,is_training=True):
    """
    This function creates the graph for the pose and explainability networks

    input:  3 consecutive frames (2 source, 1 targets), whether or not is training
    output: The mask of objects to ignore and the pose 6 Degrees of
            freedom (DoF) plus all the layers

    parameters
    ---

    target: tensor [N,W,H,3]
    sources: tensor [N,W,H,3*num_source]
    num_sources: number of sources used
    explain: whether or not create a exaplainability mask
    is_training: whether or not is training
    """
    inputs = tf.concat([target,sources],axis=3)
    _,H,W,_ = inputs.get_shape().as_list()
    #H = tf.shape(inputs)[1]
    #W = tf.shape(inputs)[2]

    with tf.variable_scope("pos_exp_net") as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(0.05),
                activation_fn=tf.nn.relu,
                outputs_collections=end_points_collection):

            # conv1 to conv5b are shared between pose and explainability prediction
            conv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='conv1')
            conv2  = slim.conv2d(conv1, 32,  [5, 5], stride=2, scope='conv2')
            conv3  = slim.conv2d(conv2, 64,  [3, 3], stride=2, scope='conv3')
            conv4  = slim.conv2d(conv3, 128, [3, 3], stride=2, scope='conv4')
            conv5  = slim.conv2d(conv4, 256, [3, 3], stride=2, scope='conv5')

            with tf.variable_scope("pose"):
                conv6  = slim.conv2d(conv5, 256, [3, 3], stride=2, scope='conv6')
                conv7  = slim.conv2d(conv6, 256, [3, 3], stride=2, scope='conv7')

                # predicting the 6DoF for each frame
                # TODO is this per stack of frames? or per frame?
                poses_pred = slim.conv2d(conv7, 6*num_sources , [1,1], stride=1,
                        normalizer_fn=None, activation_fn=None, scope='pose_pred')
                poses_avg  = tf.reduce_mean(poses_pred,[1,2])

                # Empirically the authors found that scaling by a small constant
                # facilitates training.
                poses_final = 0.01 * tf.reshape(poses_avg, [-1, num_sources, 6])

            # Exp mask specific layers
            if explain:
                with tf.variable_scope("exp"):
                    # Upscaling (Translated) convolutions
                    tconv5 = slim.conv2d_transpose(conv5,  256, [3,3], stride=2, scope="tconv5")
                    tconv4 = slim.conv2d_transpose(tconv5, 128, [3,3], stride=2, scope="tconv4")
                    tconv3 = slim.conv2d_transpose(tconv4, 64 , [3,3], stride=2, scope="tconv3")
                    tconv2 = slim.conv2d_transpose(tconv3, 32 , [5,5], stride=2, scope="tconv2")
                    tconv1 = slim.conv2d_transpose(tconv2, 16 , [7,7], stride=2, scope="tconv1")

                    # Explainability Masks
                    mask4  = slim.conv2d(tconv4, num_sources * 2, [3,3], stride=1, 
                            normalizer_fn=None, activation_fn=None, scope="mask4")
                    mask3  = slim.conv2d(tconv3, num_sources * 2, [3,3], stride=1, 
                            normalizer_fn=None, activation_fn=None, scope="mask3")
                    mask2  = slim.conv2d(tconv2, num_sources * 2, [5,5], stride=1, 
                            normalizer_fn=None, activation_fn=None, scope="mask2")
                    mask1  = slim.conv2d(tconv1, num_sources * 2, [7,7], stride=1, 
                            normalizer_fn=None, activation_fn=None, scope="mask1")
            else:
                mask4 = None
                mask3 = None
                mask2 = None
                mask1 = None

            end_points = utils.convert_collection_to_dict(end_points_collection)

    return poses_final, [mask1,mask2,mask3,mask4], end_points
                

def depth_net(target,is_training=True):
    """
    This function creates the network in charge of estimating the depth.
    This architecture is a simple convolution-deconvolution network

    input: single frame, whether is trainig or not
    output: depth for every pixel (WxH)
    """
    H = target.get_shape()[1].value
    W = target.get_shape()[2].value

    with tf.variable_scope("depth_net") as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                outputs_collections=end_points_collection,
                weights_regularizer=slim.l2_regularizer(0.05),
                activation_fn=tf.nn.relu):

            # Downscaling Convolutions
            conv1a = slim.conv2d(target,32 ,[7,7],stride=2,scope="conv1a") # downscaling
            conv1b = slim.conv2d(conv1a,32 ,[7,7],stride=1,scope="conv1b")
            conv2a = slim.conv2d(conv1b,64 ,[5,5],stride=2,scope="conv2a") # downscaling
            conv2b = slim.conv2d(conv2a,64 ,[5,5],stride=1,scope="conv2b")
            conv3a = slim.conv2d(conv2b,128,[3,3],stride=2,scope="conv3a") # downscaling
            conv3b = slim.conv2d(conv3a,128,[3,3],stride=1,scope="conv3b")
            conv4a = slim.conv2d(conv3b,256,[3,3],stride=2,scope="conv4a") # downscaling
            conv4b = slim.conv2d(conv4a,256,[3,3],stride=1,scope="conv4b")
            conv5a = slim.conv2d(conv4b,512,[3,3],stride=2,scope="conv5a") # downscaling
            conv5b = slim.conv2d(conv5a,512,[3,3],stride=1,scope="conv5b")
            conv6a = slim.conv2d(conv5b,512,[3,3],stride=2,scope="conv6a") # downscaling
            conv6b = slim.conv2d(conv6a,512,[3,3],stride=1,scope="conv6b")
            conv7a = slim.conv2d(conv6b,512,[3,3],stride=2,scope="conv7a") # downscaling
            conv7b = slim.conv2d(conv7a,512,[3,3],stride=1,scope="conv7b")

            # Upscaling (Translated) Convolutions
            tconv7a = slim.conv2d_transpose(conv7b,512,[3,3],stride=2,scope="tconv7") # upscaling
            tconv7a = resize_like(tconv7a,conv6b)
            tconv7a = tf.concat([tconv7a,conv6b],axis=3)
            tconv7b = slim.conv2d(tconv7a,512,[3,3],stride=1,scope="tconv7b")

            tconv6a = slim.conv2d_transpose(tconv7b,512,[3,3],stride=2,scope="tconv6") # upscaling
            tconv6a = resize_like(tconv6a,conv5b)
            tconv6a = tf.concat([tconv6a,conv5b],axis=3)
            tconv6b = slim.conv2d(tconv6a,512,[3,3],stride=1,scope="tconv6b")

            tconv5a = slim.conv2d_transpose(tconv6b,256,[3,3],stride=2,scope="tconv5") # upscaling
            tconv5a = resize_like(tconv5a,conv4b)
            tconv5a = tf.concat([tconv5a,conv4b],axis=3)
            tconv5b = slim.conv2d(tconv5a,256,[3,3],stride=1,scope="tconv5b")

            tconv4a = slim.conv2d_transpose(tconv5b,128,[3,3],stride=2,scope="tconv4") # upscaling
            tconv4a = tf.concat([tconv4a,conv3b],axis=3,name="concat4")
            tconv4b = slim.conv2d(tconv4a,128,[3,3],stride=1,scope="tconv4b")

            # predicting
            disp4 = MIN_DISP + DISP_SCALING * slim.conv2d(tconv4b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp4")
            # upscaling prediction
            disp4_up = tf.image.resize_bilinear(disp4,[np.int(H/4),np.int(W/4)],align_corners=True) 

            tconv3a = slim.conv2d_transpose(tconv4b,64,[3,3],stride=2,scope="tconv3") # upscaling
            tconv3a = resize_like(tconv3a,conv2b)
            tconv3a = tf.concat([tconv3a,conv2b,disp4_up],axis=3,name="concat3")
            tconv3b = slim.conv2d(tconv3a,64,[3,3],stride=1,scope="tconv3b")

            # predicting
            disp3 = MIN_DISP + DISP_SCALING * slim.conv2d(tconv3b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp3")
            # upscaling prediction
            disp3_up = tf.image.resize_bilinear(disp3,[np.int(H/2),np.int(W/2)]) 

            tconv2a = slim.conv2d_transpose(tconv3b,32,[3,3],stride=2,scope="tconv2") # upscaling
            tconv2a = resize_like(tconv2a,conv1b)
            tconv2a = tf.concat([tconv2a,conv1b,disp3_up],axis=3,name="concat2")
            tconv2b = slim.conv2d(tconv2a,32,[3,3],stride=1,scope="tconv2b")

            # predicting
            disp2 = MIN_DISP + DISP_SCALING * slim.conv2d(tconv2b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp2")
            # upscaling prediction
            disp2_up = tf.image.resize_bilinear(disp2,[H,W]) 

            tconv1a = slim.conv2d_transpose(tconv2b,16,[3,3],stride=2,scope="tconv1") # upscaling
            tconv1a = tf.concat([tconv1a,disp2_up],axis=3,name="concat1")
            tconv1b = slim.conv2d(tconv1a,16,[3,3],stride=1,scope="tconv1b")

            # predicting
            disp1 = MIN_DISP + DISP_SCALING * slim.conv2d(tconv1b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp1")

            end_points = utils.convert_collection_to_dict(end_points_collection)

    return [disp1,disp2,disp3,disp4], end_points

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])
