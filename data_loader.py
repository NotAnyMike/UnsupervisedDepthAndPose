import os
import random
import tensorflow as tf

class DataLoader():
    """
    Takes care of loading and augmenting the dataset and
    related operations
    """
    def __init__(self,
            dataset_dir,
            batch_size,
            num_source,
            num_scale,
            seed,):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_source = num_source
        self.num_scale = num_scale
        self.seed = seed

    def load_train_batch(self):
        """
        Load a batch for training
        """

        # Getting list of all files for training
        img_paths,cam_paths = self.format_file_list(self.dataset_dir,'train')
        
        # Shuffling lists
        img_paths_queue = tf.random.shuffle(img_paths,seed=self.seed)
        cam_paths_queue = tf.random.shuffle(cam_paths,seed=self.seed)

        self.steps_per_epoch = int(len(img_paths//self.batch_size))

        # Load imgs
        tf.data.

    def format_file_list(self, data_root, split):
        """
        reads the file for training or val set (split) in the data_root


        data_root: The root (main) folder where the data is
        split: Whether 'train' or 'val', the type of set we are 
        going to use
        """
        with open(os.path.join(data_root,'%s.txt' % split),'r') as f:
            frames = f.readlines()

        # Data is list of lists, each list contains the subfolder 
        # of the image (the name of the sequence from where the
        # image was taken) and the id of the frame/image
        data = [x.split() for in frames]

        img_file_list = [os.path.join(data_root,subfolder,id) + '.jpg' \
                for subfolder,id in data]

        cam_file_list = [os.path.join(data_root,subfolder,id) + '_cam.txt' \
                for subfolder,id in data]

        return img_file_list,cam_file_list
