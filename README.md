# SfMLearner Updated

Implementation of Unsupervised Learning of Depth and Ego-Motion from Video implemented with the more updated 10.1 CUDA, 418.87 Nvidia driver and tensorflow-gpu 1.14 in Ubuntu 18.04. This repo diverges slightly from the original repo [SfMLearner](https://github.com/tinghuiz/SfMLearner), there are also the pytorch implementation [here](https://github.com/ClementPinard/SfmLearner-Pytorch)

The objective of this repository was to replicate the original paper in order to practice and learn, not for research purposes. This model was the first one using this method. Currently there are several improved models.

## Preliminary Outputs

Some of the outputs from the model:

| ![2011_09_26_drive_0005_sync_02](misc/2011_09_26_drive_0005_sync_02.gif) | ![Sequence 21](misc/sequences_21.gif) |
| ------------------------------------------------------------ | ------------------------------------- |
| ![Sequence 06](misc/sequences_06.gif)                        | ![Sequence 07](misc/sequences_07.gif) |

Those results are only preliminary. I haven't evaluated and compared the results to the original paper yet.

## Differences

There are few differences, the most important is that all the code is properly commented to guide anyone. There are more subtle differences such as

* The model input dataset for training is using the `Dataset` object from `tf.data` which makes the model a bit more efficient.
* The model can be easily trained with any kind of images of random constant size.
* Inference can take any number of images (limited by memory) and is only one function.

Some functions have been taken from the original paper without any change.

## Results

### Depth

#### Evaluation 

In order to evaluate the model, first generate the predictions for the files from the kitti eigen split you find [here](data/kitti/test_files_eigen.txt). To generate it you can use the notebook for that purpose. One you can the predictions run

```
python kitti_eval/eval_depth.py --kitti_dir /HDD/Downloads/Datasets/raw_data_downloader/ --pred_file=kitti_eval/kitti_eigen_depth_predictions.npy
```

as in the original code.

#### Metrics

| Model               | Absolute Relative Error | Square Relative Error |   RMS   | Log RMS | d1_all |   a1   |   a2   |   a3   |
| :------------------ | :---------------------: | :-------------------: | :-----: | :-----: | :----: | :----: | :----: | :----: |
| SfMLearner Original |         0.1978          |        1.8363         | 6.5645  | 0.2750  | 0.000  | 0.7176 | 0.9010 | 0.9606 |
| This model          |         0.4614          |        5.4931         | 12.3904 | 0.6104  | 0.000  | 0.3036 | 0.5573 | 0.7446 |

### Pose

TODO

## Instructions

Here are the instructions to use the model for inference or to train it.

### Inference

To run inference using the model, you will need the trained weights from [here](https://drive.google.com/file/d/12qKGizia5jJhqLm0UDXlVBSoeI5dmgmI/view?usp=sharing)

1. It is better to have a conda environment to simplify the GPU configuration, but this is optional
	
	1. `conda create -n SfMLearner python=3.7 tensorflow-gpu` and confirm everything
	1. `source activate SfMLearner`

1. Install requirements `pip install -r requirements`
2. If you are not using conda, then install the version of tensorflow that suites you.
3. Run jupyter `jupyter notebook`
4. Open the notebook `Inference.ipynb` and adjust the parameters to your case (the weights and the images you want to use) (have in mind that if you trained the model with different height and width, the inference function takes those two arguments as inputs as well).
5. Run entire notebook
6. To repeat, reset the kernel to free memory in the GPU and re-run the notebook.

### Training

Install the requirements and the environment as in the inference first two steps. Next it is necessary to have the data-set pre-processed.

#### 1. Downloading dataset

In order to keep as close to the original paper, this repo uses the same pre-process pipeline from the original paper. So far I have only been limited to the [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset, download it using the following script from them ([here](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)).

#### 2. Preprocess

Once downloaded the data set, it is necessary to arrange it in the format the model will use. The pipeline here is exactly the same as the original paper.

run: `python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4`

#### 2. Train

1. Activate the environment with `source activate SfMLearner` (if you are using one).
2. Train the model with `python train.py --dataset_dir=/HDD/Documents/SfMLearner_data --checkpoint_dir=chpts_2m_nomask --batch_size=4 --smooth_weight 2.0`. You can configure training further with the instructions from `train.py` as `python train.py --help`.
3. This will run for around 20 epochs, the results will be saved in `checkpoint_dir`.

## Notes

This repo has only learning purposes