from pdb import set_trace

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
            num_sources,
            num_scales,
            num_parallel,
            seed,):

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_sources = num_sources
        self.num_scales = num_scales
        self.num_parallel = num_parallel
        self.seed = seed

    def load_batch(self,split):
        """
        Loads a batch for training. Returns an iterator, to get the elment
        use iterator.get_next(). This class reads the file for training
        or val set (split) in the data_root. each element will return a list
        of bath_size points, each containing their corresponding
        target, sources, intrinsic matrix and data. Data Contains
        the subfolder of the frames and the number of the frames.
        Data is not used in training and can be ignored.

        split: Whether 'train' or 'val', the type of set we are 
        going to use
        """

        with open(os.path.join(self.dataset_dir,split+".txt")) as f:
            self.total_imgs = len(f.readlines())
            self.steps_per_epoch = int(self.total_imgs // self.batch_size)

        # f,i,d stands for Frame, Intrinsics, Data
        imgs = tf.data.TextLineDataset(os.path.join(self.dataset_dir,split+".txt"))
        # A perfect shuffle has the same lenght as the whole data 
        imgs = imgs.shuffle(self.total_imgs, seed=self.seed) 
        imgs = imgs.map(lambda string: tf.string_split([string]).values)
        imgs = imgs.map(lambda d : self._parse_fn(d[0],d[1]), num_parallel_calls=self.num_parallel)
        imgs = imgs.map(lambda f,i,d : self._separate_fn(f,i,d), num_parallel_calls=self.num_parallel)
        imgs = imgs.map(lambda f,i,d : self._data_augmentation(f,i,d), num_parallel_calls=self.num_parallel)
        imgs = imgs.map(lambda f,i,d : self._scale_intrinsics_fn(f,i,d), num_parallel_calls=self.num_parallel)
        imgs = imgs.map(lambda f,i,d : self._separate_target_fn(f,i,d), num_parallel_calls=self.num_parallel)
        imgs = imgs.batch(self.batch_size,drop_remainder=True)
        imgs = imgs.repeat()
        imgs = imgs.prefetch(1)

        if tf.executing_eagerly():
            return imgs
        else:
            iterator = imgs.make_initializable_iterator()
            return iterator

    def _parse_fn(self,subfolder,file):
        """
        Reads the jpe image as well as the intrinsic matrix file
        """
        img_str = tf.io.read_file(tf.strings.join([self.dataset_dir,subfolder,file+'.jpg'],'/'))
        intrinsics = tf.io.read_file(tf.strings.join([self.dataset_dir,subfolder,file+'_cam.txt'],'/'))

        intrinsics = tf.string_split([intrinsics],',').values
        intrinsics = tf.strings.to_number(intrinsics)
        intrinsics = tf.reshape(intrinsics, [3,3])

        img = tf.image.decode_jpeg(img_str,channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [128, 1248])

        return img,intrinsics,[subfolder,file]

    def _scale_intrinsics_fn(self,frames,intrinsics,data):
        multiscale_intrinsics = [intrinsics]
        for scale in range(self.num_scales-1):
            tmp_intrinsics = tf.unstack(tf.reshape(intrinsics,[-1]))

            tmp_intrinsics[0*3+0] /= (2 ** scale) # fx
            tmp_intrinsics[1*3+1] /= (2 ** scale) # fy
            tmp_intrinsics[0*3+2] /= (2 ** scale) # cx
            tmp_intrinsics[1*3+2] /= (2 ** scale) # cy

            tmp_intrinsics = tf.stack(tmp_intrinsics)
            tmp_intrinsics = tf.reshape(tmp_intrinsics, [3,3])

            multiscale_intrinsics.append(tmp_intrinsics)

        return frames,multiscale_intrinsics,data

    def _separate_fn(self,frames,intrinsics,data):
        img_width = int(frames.shape[1]//(self.num_sources+1))
        img_height= frames.shape[0]

        # Target frame
        target = tf.slice(
            frames,
            [0,img_width*(self.num_sources//2),0],
            [-1,img_width,-1])

        # Source frames
        src1 = tf.slice(
            frames,
            [0,0,0],
            [-1,img_width,-1])
        src2 = tf.slice(
            frames,
            [0,2*img_width,0],
            [-1,img_width,-1])

        # The order is to allow random changes in images easier
        frames = tf.concat([src1,src2,target],axis=2)
        frames.set_shape([img_height,img_width,3*(self.num_sources+1)])
        return frames,intrinsics,data

    def _separate_target_fn(self,frames,intrinsics,data):
        """
        Returns the target frame, source frames and intrisics.
        We asume the target is at last in the color channel
        """
        img_height,img_width,c = frames.shape

        # Target frame
        target = tf.slice(
            frames,
            [0,0,tf.cast(c-3,tf.int32)],
            [-1,-1,3])

        # Source frames
        srcs = tf.slice(
            frames,
            [0,0,0],
            [-1,-1,6])

        return target,srcs,intrinsics,data


    def _data_augmentation(self,img,intrinsics,data):
        img_height,img_width,_ = img.get_shape().as_list()


        # Random scaling
        def random_scaling(img, intrinsics):
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]

            new_h = tf.cast(img_height * y_scaling, dtype=tf.int32)
            new_w = tf.cast(img_width  * x_scaling, dtype=tf.int32)

            img = tf.image.resize(img,[new_h,new_w])

            intrinsics[0*3+0] *= x_scaling # fx
            intrinsics[1*3+1] *= y_scaling # fy
            intrinsics[0*3+2] *= x_scaling # cx
            intrinsics[1*3+2] *= y_scaling # cy

            return img, intrinsics

        # Random cropping
        def random_cropping(img,intrinsics):
            #new_h,new_w,_ = img.get_shape().as_list()
            new_h = tf.shape(img)[0]
            new_w = tf.shape(img)[1]

            offset_y = tf.random.uniform([1], 0, new_h - img_height + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, new_w - img_width  + 1, dtype=tf.int32)[0]

            img = tf.image.crop_to_bounding_box(
                img, offset_y, offset_x, img_height, img_width)

            intrinsics[0*3+2] -= tf.cast(offset_x, dtype=tf.float32) # cx
            intrinsics[1*3+2] -= tf.cast(offset_y, dtype=tf.float32) # cy

            return img, intrinsics

        intrinsics = tf.unstack(tf.reshape(intrinsics,[-1]))

        img,intrinsics = random_scaling(img,intrinsics)
        img,intrinsics = random_cropping(img,intrinsics)
        img = tf.cast(img, dtype=tf.float32)

        intrinsics = tf.stack(intrinsics)
        intrinsics = tf.reshape(intrinsics,[3,3])

        return img, intrinsics, data
