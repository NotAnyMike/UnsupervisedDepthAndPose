import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils

MIN_DISP = 0.01
DISP_SCALING = 10

def pos_exp_net(img1,img2,img3,explain=True,is_training=True):
    """
    This function creates the graph for the pose and explainability networks

    input:  3 consecutive frames, whether or not is training
    output: The mask of objects to ignore and the pose 6 Degrees of freedom (DoF)
    """
    imgs = tf.concat([img1,img2,img3],axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value

    # TODO Understand what this is exactly
    # The number of contiguous frames or num of stack of frames?
    num_source = int(src_image_stack.get_shape()[3].value//3)

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
                conv7  = slim.conv2d(conv7, 256, [3, 3], stride=2, scope='conv7')

                # predicting the 6DoF for each frame
                # TODO is this per stack of frames? or per frame?
                pose_pred = slim.conv2d(conv7, 6*num_source , [1,1], stride=1,
                        normalizer_fn=None, activation_fn=None, scope='pose_pred')
                pose_avg  = tf.reduce_mean(pose_pred,[1,2])
                

def depth_net(img,is_training=True):
    """
    This function creates the network in charge of estimating the depth.
    This architecture is a simple convolution-deconvolution network

    input: single frame, whether is trainig or not
    output: depth for every pixel (WxH)
    """
    H = img.get_shape()[1].value
    W = img.get_shape()[2].value

    with tf.scope("depth_net") as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d.conv2d_transpose],
                outputs_collections=end_points_collection,
                weights_regularizer=slim.l2_regularizer(0.05),
                activation_fn=tf.nn.relu):

            # Downscaling CONVolutions
            conv1a = slim.conv2d(img    ,32 ,[7,7],stride=2,scope="conv1a") # downscaling
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

            # Upscaling CONVolutions
            tcont7a = slim.conv2d_transpose(conv7b,512,[3,3],stride=2,scope="tcont7") # upscaling
            tcont7a = tf.concat([tcont7a,conv6b],axis=3)
            tcont7b = slim.conv2d(tcont7a,512,[3,3],stride=1,scope="tcont7b")

            tcont6a = slim.conv2d_transpose(tcont7b,512,[3,3],stride=2,scope="tcont6") # upscaling
            tcont6a = tf.concat([tcont6a,conv5b],axis=3)
            tcont6b = slim.conv2d(tcont6a,512,[3,3],stride=1,scope="tcont6b")

            tcont5a = slim.conv2d_transpose(tcont6b,256,[3,3],stride=2,scope="tcont5") # upscaling
            tcont5a = tf.concat([tcont5a,conv4b],axis=3)
            tcont5b = slim.conv2d(tcont5a,256,[3,3],stride=1,scope="tcont5b")

            tcont4a = slim.conv2d_transpose(tcont5b,128,[3,3],stride=2,scope="tcont4") # upscaling
            tcont4a = tf.concat([tcont4a,conv3b],axis=3)
            tcont4b = slim.conv2d(tcont4a,128,[3,3],stride=1,scope="tcont4b")

            # predicting
            disp4 = MIN_DISP + DISP_SCALING * slim.conv2d(tcont4b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp4")
            # upscaling prediction
            disp4_up = tf.image.resize_bilinear(disp4,[np.int(H/4),np.int(W/4)]) 

            tcont3a = slim.conv2d_transpose(tcont4b,64,[3,3],stride=2,scope="tcont3") # upscaling
            tcont3a = tf.concat([tcont3a,conv2b,disp4_up],axis=3)
            tcont3b = slim.conv2d(tcont3a,64,[3,3],stride=1,scope="tcont3b")

            # predicting
            disp3 = MIN_DISP + DISP_SCALING * slim.conv2d(tcont3b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp3")
            # upscaling prediction
            disp3_up = tf.image.resize_bilinear(disp3,[np.int(H/2),np.int(W/2)]) 

            tcont2a = slim.conv2d_transpose(tcont3b,32,[3,3],stride=2,scope="tcont2") # upscaling
            tcont2a = tf.concat([tcont2a,conv1b,disp3_up],axis=3)
            tcont2b = slim.conv2d(tcont2a,32,[3,3],stride=1,scope="tcont2b")

            # predicting
            disp2 = MIN_DISP + DISP_SCALING * slim.conv2d(tcont2b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp2")
            # upscaling prediction
            disp2_up = tf.image.resize_bilinear(disp2,[H,W]) 

            tcont1a = slim.conv2d_transpose(tcont2b,16,[3,3],stride=2,scope="tcont1") # upscaling
            tcont1a = tf.concat([tcont1a,disp2_up],axis=3)
            tcont1b = slim.conv2d(tcont1a,16,[3,3],stride=1,scope="tcont1b")

            # predicting
            disp1 = MIN_DISP + DISP_SCALING * slim.conv2d(tcont1b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp1")

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1,disp2,disp3,disp4], end_points
