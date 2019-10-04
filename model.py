import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib
import PIL.Image as pil

from data_loader import DataLoader
from nets import depth_net,pos_exp_net
from utils import projective_inverse_warp

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
            continue_training,
            init_checkpoint_file,
            learning_rate,
            beta1,
            smooth_weight,
            mask_weight,
            exp_regularization_weight,
            batch_size,
            max_steps,
            summary_freq,
            save_model_freq,
            num_parallel,
            seed,
            ):

        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.continue_training = continue_training
        self.init_checkpoint_file = init_checkpoint_file
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.summary_freq = summary_freq
        self.save_model_freq = save_model_freq
        self.num_parallel = num_parallel
        self.seed = seed
        
        # weights
        self.smooth_weight = smooth_weight
        self.mask_weight = mask_weight
        self.exp_regularization_weight = exp_regularization_weight

    def _build_graphs(self):
        """
        Function in charge of building the 3 main graphs and
        computing the loss. The main graphs are the depth, 
        pose and explainability mask
        """
        # Load data
        loader = DataLoader(
                dataset_dir=self.dataset_dir,
                batch_size   = self.batch_size,
                num_parallel = self.num_parallel,
                num_sources  = NUM_SOURCES,
                num_scales   = NUM_SCALES,
                seed         = self.seed,
                )

        with tf.name_scope("data_loading"):
            dataset = loader.load_batch('train')
            target,sources,intrinsics,_  = dataset.get_next()
            target = self._preprocess_img(target)
            sources= self._preprocess_img(sources)

        with tf.name_scope("depth_prediction"):
            depths, depth_end_points = depth_net(target,is_training=True)
            depths = [1./d for d in depths] # d is between 0 and 1
        
        with tf.name_scope("pose_and_explainability_prediction"):
            poses, masks, pose_and_exp_end_points = \
                    pos_exp_net(
                            target,
                            sources,
                            num_sources=NUM_SOURCES,
                            explain=True,
                            is_training=True)
        
        with tf.name_scope("compute_loss"):
            # Losses
            pixel_loss = 0  # Difference in pixels
            smooth_loss = 0 # For smoothness
            exp_loss = 0    # Explainability mask loss

            # List to save images and masks
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []

            _,height,width,_ = target.get_shape().as_list()
            total_channels = tf.shape(sources)[-1]
            print(target.shape)
            print(height)

            # For each scale
            for scale in range(NUM_SCALES):
                # Calculating dimensions for current scale
                curr_dims = []
                curr_dims.append(tf.cast(height / (2**scale),tf.int32))
                curr_dims.append(tf.cast(width  / (2**scale),tf.int32))

                # Resizing targets and source images
                curr_target  = tf.image.resize(target, curr_dims)
                curr_sources = tf.image.resize(sources,curr_dims)

                if self.exp_regularization_weight > 0:
                    # Construct a reference explainability mask (i.e. all
                    # pixels are explainable)
                    # This will be the goal of the explainibility mask, in order
                    # to avoid converging to a mask with all zeros
                    ref_exp_mask = self.get_reference_explain_mask(scale,height,width)

                # Calculate smooth loss for all scales
                if self.smooth_weight > 0:
                    curr_smooth_loss = self._compute_smooth_loss(depths[scale])
                    smooth_loss += self.smooth_weight/(2**scale)*curr_smooth_loss

                # For each source
                for source_idx in range(NUM_SOURCES):
                    # Inverse wrap the source to the target
                    # Channels needed in case different number of channels, eg. B&W
                    channels = tf.cast(total_channels / NUM_SOURCES,dtype=tf.int32)
                    from_c = channels * source_idx
                    to_c   = channels * (source_idx+1)
                    curr_source = curr_sources[:,:,:,from_c:to_c]

                    projected_img = projective_inverse_warp(
                            curr_source,
                            tf.squeeze(depths[scale], axis=3),
                            poses[:,source_idx,:],
                            intrinsics[:,source_idx,:,:],
                            self.batch_size)

                    # Reconstruction error
                    curr_proj_error = tf.abs(projected_img - curr_target)

                    # Cross-entropy loss as regularization for the
                    # explainability prediction
                    if self.exp_regularization_weight > 0:
                        curr_exp_logits = tf.slice(masks[scale],
                                [0, 0, 0, source_idx*2],
                                [-1, -1, -1, 2])

                        # To avoid overfitting to mask full of zeros to avoid
                        # explaining everything
                        curr_exp_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=tf.reshape(curr_exp_logits,[-1,2]),
                                labels=tf.reshape(ref_exp_mask,[-1,2]))
                        curr_exp_loss = self.exp_regularization_weight * \
                                tf.reduce_mean(curr_exp_loss)

                        exp_loss += curr_exp_loss
                        curr_exp = tf.nn.softmax(curr_exp_logits) # Explainability mask

                    # Photo-consistency loss weighted by explainability
                    if self.exp_regularization_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                                tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error)

                    # Prepare images for tensorboard summaries
                    if source_idx == 0:
                        proj_image_stack = projected_img
                        proj_error_stack = curr_proj_error
                        if self.exp_regularization_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack,
                                                      projected_img], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                        if self.exp_regularization_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack,
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)

                tgt_image_all.append(curr_target)
                src_image_stack_all.append(curr_sources)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if self.exp_regularization_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
                        

            # Total loss
            total_loss = pixel_loss + smooth_loss + exp_loss
        
        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)

            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.dataset = dataset
        self.depths = depths
        self.poses = poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def _collect_summaries(self):
        """
        Creates the necessary summaries for the training
        """
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        
        for scale in range(NUM_SCALES):
            tf.summary.histogram("scale%d_depth" % scale, self.depths[scale])
            tf.summary.image('scale%d_disparity_image' % scale, 1./self.depths[scale])
            tf.summary.image('scale%d_target_image' % scale, \
                             self._deprocess_image(self.tgt_image_all[scale]))
            for idx_source in range(NUM_SOURCES):
                if self.exp_regularization_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (scale, idx_source),
                        tf.expand_dims(self.exp_mask_stack_all[scale][:,:,:,idx_source], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (scale, idx_source),
                    self._deprocess_image(self.src_image_stack_all[scale][:, :, :, idx_source*3:(idx_source+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (scale, idx_source),
                    self._deprocess_image(self.proj_image_stack_all[scale][:, :, :, idx_source*3:(idx_source+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (scale, idx_source),
                    self._deprocess_image(tf.clip_by_value(
                        self.proj_error_stack_all[scale][:,:,:,idx_source*3:(idx_source+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.poses[:,:,0])
        tf.summary.histogram("ty", self.poses[:,:,1])
        tf.summary.histogram("tz", self.poses[:,:,2])
        tf.summary.histogram("rx", self.poses[:,:,3])
        tf.summary.histogram("ry", self.poses[:,:,4])
        tf.summary.histogram("rz", self.poses[:,:,5])


    def train(self):
        self._build_graphs()
        self._collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])

        # To save checkpoints
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)

        # Necessary
        #sv = tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
        #                         save_summaries_secs=0)

        summary_op = tf.summary.merge_all()
        latestSaverHook = tf.train.CheckpointSaverHook(
                self.checkpoint_dir,
                save_steps=self.save_model_freq,
                checkpoint_basename='model.latest')
        gsSaverHook = tf.train.CheckpointSaverHook(
                self.checkpoint_dir,
                save_steps=self.steps_per_epoch,
                checkpoint_basename='model')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #with sv.managed_session(config=config) as sess:
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.checkpoint_dir,
                save_summaries_secs=0,
                config=config,
                hooks=[latestSaverHook,gsSaverHook]) as sess:
            summary_writer = sess._hooks[1]._summary_writer
            #summary_op = sess._hooks[1]._summary_op

            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            if self.continue_training:
                if self.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                else:
                    checkpoint = self.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()

            for step in range(1, int(self.max_steps)):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % self.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    #fetches["summary"] = sv.summary_op
                    fetches["summary"] = summary_op

                sess.run(self.dataset.initializer)
                results = sess.run(fetches)
                gs = results["global_step"]

                if step % self.summary_freq == 0:
                    #sv.summary_writer.add_summary(results["summary"], gs)
                    summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/self.summary_freq,
                                results["loss"]))
                    start_time = time.time()

                #if step % self.save_latest_freq == 0:
                #if step % self.save_model_freq == 0:
                    #self.save(sess, self.checkpoint_dir, 'latest')

                #if step % self.steps_per_epoch == 0:
                    #self.save(sess, self.checkpoint_dir, gs)
        print("Done training")

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

    def _deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def get_reference_explain_mask(self, downscaling,img_height,img_width):
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (self.batch_size,
                                int(img_height/(2**downscaling)),
                                int(img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    def _build_depth_test_graph(self,height,width):
        input_raw = tf.placeholder(tf.uint8, 
                [self.batch_size, height,width,3], name="raw_input")
        input_mc  = self._preprocess_img(input_raw)

        with tf.name_scope("depth_prediction"):
            depths, depth_end_points = depth_net(input_mc,is_training=False)
            depths = [1./d for d in depths]

        return depths, input_raw, depth_end_points

    def infere_depth(self,img_path,checkpoint_model):
        """
        Args:
            images_location: list of the files to load, all images have to have
                             the same height and width
        Returns: depths for different scales specified in training
        """
        # Load images and get height,width
        img_height = 128
        img_width  = 416
        img = pil.open(img_path)
        img = img.resize((img_width, img_height), pil.ANTIALIAS)
        img = np.array(img)

        # Build depth test graph
        depths, input_ph, depth_end_points = \
                self._build_depth_test_graph(img_height,img_width)

        # Create saver
        saver = tf.train.Saver([var for var in tf.model_variables()])

        # Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Recover variable values
            saver.restore(sess,checkpoint_model)

            # Predict
            feed_dict = {input_ph:img[None,:,:,:]}
            fetches = [depths]
            pred = sess.run(fetches,feed_dict=feed_dict)

        # Return prediction
        return pred[0]
