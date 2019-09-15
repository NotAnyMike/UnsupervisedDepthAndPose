import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import matplotlib

# Constants
SEQ_LENGTH = 3
NUM_SOURCE = 2
NUM_SCALES = 4

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
            ):

        self.num_seq = 3 # Fixed

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

    def build_graphs(self):
        """
        Function in charge of building the 3 main graphs and
        computing the loss. The main graphs are the depth, 
        pose and explainability mask
        """
        # Create the loader
        # TODO
        with tf.name_scope("data_loading"):
            pass
        with tf.name_scope("depth_prediction"):
            pass
        with tf.name_scope("pose_and_explainability_prediction"):
            pass
        with tf.name_scope("compute_loss"):
            pass
        with tf.name_scope("train_op"):
            pass

        tf.image
    
    def collect_summaries(self):
        """
        Creates the necessary summaries for the training
        """
        pass

    def train(self):
        self.build_graphs()
        self.collect_summaries()
