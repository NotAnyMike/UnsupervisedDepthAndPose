import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils

MIN_DISP = 0.01
DISP_SCALING = 10

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
            dconv1a = slim.conv2d(img    ,32 ,[7,7],stride=2,scope="dconv1a") # downscaling
            dconv1b = slim.conv2d(dconv1a,32 ,[7,7],stride=1,scope="dconv1b")
            dconv2a = slim.conv2d(dconv1b,64 ,[5,5],stride=2,scope="dconv2a") # downscaling
            dconv2b = slim.conv2d(dconv2a,64 ,[5,5],stride=1,scope="dconv2b")
            dconv3a = slim.conv2d(dconv2b,128,[3,3],stride=2,scope="dconv3a") # downscaling
            dconv3b = slim.conv2d(dconv3a,128,[3,3],stride=1,scope="dconv3b")
            dconv4a = slim.conv2d(dconv3b,256,[3,3],stride=2,scope="dconv4a") # downscaling
            dconv4b = slim.conv2d(dconv4a,256,[3,3],stride=1,scope="dconv4b")
            dconv5a = slim.conv2d(dconv4b,512,[3,3],stride=2,scope="dconv5a") # downscaling
            dconv5b = slim.conv2d(dconv5a,512,[3,3],stride=1,scope="dconv5b")
            dconv6a = slim.conv2d(dconv5b,512,[3,3],stride=2,scope="dconv6a") # downscaling
            dconv6b = slim.conv2d(dconv6a,512,[3,3],stride=1,scope="dconv6b")
            dconv7a = slim.conv2d(dconv6b,512,[3,3],stride=2,scope="dconv7a") # downscaling
            dconv7b = slim.conv2d(dconv7a,512,[3,3],stride=1,scope="dconv7b")

            # Upscaling CONVolutions
            uconv7a = slim.conv2d_transpose(dconv7b,512,[3,3],stride=2,scope="uconv7") # upscaling
            uconv7a = tf.concat([uconv7a,dconv6b],axis=3)
            uconv7b = slim.conv2d(uconv7a,512,[3,3],stride=1,scope="uconv7b")

            uconv6a = slim.conv2d_transpose(uconv7b,512,[3,3],stride=2,scope="uconv6") # upscaling
            uconv6a = tf.concat([uconv6a,dconv5b],axis=3)
            uconv6b = slim.conv2d(uconv6a,512,[3,3],stride=1,scope="uconv6b")

            uconv5a = slim.conv2d_transpose(uconv6b,256,[3,3],stride=2,scope="uconv5") # upscaling
            uconv5a = tf.concat([uconv5a,dconv4b],axis=3)
            uconv5b = slim.conv2d(uconv5a,256,[3,3],stride=1,scope="uconv5b")

            uconv4a = slim.conv2d_transpose(uconv5b,128,[3,3],stride=2,scope="uconv4") # upscaling
            uconv4a = tf.concat([uconv4a,dconv3b],axis=3)
            uconv4b = slim.conv2d(uconv4a,128,[3,3],stride=1,scope="uconv4b")

            # predicting
            disp4 = MIN_DISP + DISP_SCALING * slim.conv2d(uconv4b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp4")
            # upscaling prediction
            disp4_up = tf.image.resize_bilinear(disp4,[np.int(H/4),np.int(W/4)]) 

            uconv3a = slim.conv2d_transpose(uconv4b,64,[3,3],stride=2,scope="uconv3") # upscaling
            uconv3a = tf.concat([uconv3a,dconv2b,disp4_up],axis=3)
            uconv3b = slim.conv2d(uconv3a,64,[3,3],stride=1,scope="uconv3b")

            # predicting
            disp3 = MIN_DISP + DISP_SCALING * slim.conv2d(uconv3b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp3")
            # upscaling prediction
            disp3_up = tf.image.resize_bilinear(disp3,[np.int(H/2),np.int(W/2)]) 

            uconv2a = slim.conv2d_transpose(uconv3b,32,[3,3],stride=2,scope="uconv2") # upscaling
            uconv2a = tf.concat([uconv2a,dconv1b,disp3_up],axis=3)
            uconv2b = slim.conv2d(uconv2a,32,[3,3],stride=1,scope="uconv2b")

            # predicting
            disp2 = MIN_DISP + DISP_SCALING * slim.conv2d(uconv2b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp2")
            # upscaling prediction
            disp2_up = tf.image.resize_bilinear(disp2,[H,W]) 

            uconv1a = slim.conv2d_transpose(uconv2b,16,[3,3],stride=2,scope="uconv1") # upscaling
            uconv1a = tf.concat([uconv1a,disp2_up],axis=3)
            uconv1b = slim.conv2d(uconv1a,16,[3,3],stride=1,scope="uconv1b")

            # predicting
            disp1 = MIN_DISP + DISP_SCALING * slim.conv2d(uconv1b,1,[3,3],stride=1,
                    activation_fn=tf.sigmoid,normalizer_fn=None,scope="disp1")

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1,disp2,disp3,disp4], end_points
