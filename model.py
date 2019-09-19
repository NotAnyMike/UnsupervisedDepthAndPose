import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import matplotlib

from data_loader import DataLoader
from nets import depth_net,pos_exp_net

# Constants
SEQ_LENGTH  = 3
NUM_SCALES  = 4
NUM_SOURCES = 2

class Model():
    """
    Main model of the paper Unsupervised Learning \
    of Depth and Ego-Motion from Video. For details 
    on the input parameters, run this file with 
    the flag --help or read below:


    """
    def __init__(self,
            dataset_dir,
            checkpoint_dir,
            init_checkpoint_file,
            learning_rate,
            beta1,
            smooth_weight,
            mask_weight,
            batch_size,
            max_steps,
            summary_freq,
            save_model_freq,
            num_parallel
            ):

        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.init_checkpoint_file = init_checkpoint_file
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.smooth_weight = smooth_weight
        self.mask_weight = mask_weight
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.summary_freq = summary_freq
        self.save_model_freq = save_model_freq
        self.num_parallel = num_parallel

    def _build_graphs(self):
        """
        Function in charge of building the 3 main graphs and
        computing the loss. The main graphs are the depth, 
        pose and explainability mask
        """
        # Load data
        loader = DataLoader(
                dataset_dir=,
                batch_size   = self.batch_size,
                num_parallel = self.num_parallel
                num_sources  = NUM_SOURCES,
                num_scales   = NUM_SCALES,
                seed         = self.seed,
                )

        with tf.name_scope("data_loading"):
            target,src,intrinsics,_ = loader.load_batch('train')
            target = self._preprocess_img(target)
            src    = self._preprocess_img(src)

        with tf.name_scope("depth_prediction"):
            depths, depth_end_points = depth_net(target,is_training=True)
            depths = [1./d for d in depths] # TODO WHY?
        
        with tf.name_scope("pose_and_explainability_prediction"):
            poses, masks, pose_and_exp_end_points = \
                    pos_exp_net(
                            target,
                            sources,
                            num_sources=self.num_sources,
                            explain=True,
                            is_training=True)
        
        with tf.name_scope("compute_loss"):
            # Losses
            pixel_loss = 0  # Difference in pixels
            smooth_loss = 0 # For smoothness
            exp_loss = 0    # Explainability mask loss

            _,height,width,_ = tf.shape(target)

            # For each scale
            for scale in range(self.num_scales):
                # Calculating dimensions for current scale
                curr_dims = []
                curr_dims.append(tf.cast(height / (2**scale),tf.int32))
                curr_dims.append(tf.cast(width  / (2**scale),tf.int32))

                # Resizing targets and source images
                curr_target  = tf.image.resize(target, curr_dims)
                curr_sources = tf.image.resize(sources,curr_dims)

                # Calculate smooth loss for all scales
                if self.smooth_weight > 0:
                    curr_smooth_loss = self._compute_smooth_loss(depths[s])
                    smooth_loss += self.smooth_weight/(2**scale)*curr_smooth_loss

                # For each source
                for source_idx for range(self.num_sources):
                    # Inverse wrap the source to the target
                    # Channels needed in case different number of channels, eg. B&W
                    channels = tf.cast(total_channels / self.num_sources,dtype=tf.int32)
                    from = channels * i
                    to   = channels * (i+1)
                    curr_source = curr_sources[:,:,:,from:to]

                    projected_img = self._projective_inverse_warp(
                            curr_source,
                            tf.squeeze(depth[s], axis=3),
                            poses[:,source_idx,:],
                            intrinsics[:,source_idx,:,:])

                    # TODO CONTINUE: implement projective_inverse_wrap
        
        with tf.name_scope("train_op"):
            pass

    def _collect_summaries(self):
        """
        Creates the necessary summaries for the training
        """
        pass

    def train(self):
        self._build_graphs()
        self._collect_summaries()

    def _preprocess_img(self,img):
        """
        normalizes the image between -1 and 1
        """
        img = tf.image.convert_image_dtype(img,dtype=tf.float32)
        img = img * 2. - 1.
        return img

    def _compute_smooth_loss(self,depth):
        """
        Computes the smooth loss. This function is exactly the same as
        the original paper. See above equation 4 in org paper for a small
        explanaition
        """
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(depth)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def _projective_inverse_warp(self,source,depth,pose,intrinsics):
        pass
