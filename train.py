import argparse
import numpy as np
import os

from model import Model

parser = argparse.ArgumentParser(description='Process to run the \
        main trainig operations for the paper Unsupervised \
        Learning of Depth and Ego-Motion from Video')

parser.add_argument('--dataset_dir', default="dataset/", type=str, 
        help="Directory of the dataset")
parser.add_argument('--checkpoint_dir', default="checkpoin/", type=str, 
        help="Directory to save the checkpoints from training")
parser.add_argument('--init_checkpoint_file', default=None, type=str, 
        help="Previous checkpoint file to start training from")
parser.add_argument('--learning_rate', '-lr', default=2e-4, type=float,
        help="Learning rate for the model, default 0.004")
parser.add_argument('--beta1', default=0.9, type=float,
        help="Momentum weight for Adam, default 0.9")
parser.add_argument('--smooth_weight', default=0.5, type=float,
        help="Smooth weight for loss")
parser.add_argument('--mask_weight', default=0.5, type=float,
        help="Explainability mask weight for regularization")
parser.add_argument('--batch_size', default=4, type=int,
        help="Size of the batch")
parser.add_argument('--max_steps', default=2e4, type=int,
        help="Max number of steps to train for")
parser.add_argument('--summary_freq', default=100, type=int,
        help="Frequency of the summary in number of interations")
parser.add_argument('--num_parallel', default=100, type=int,
        help="Number of parallel process in CPU (~ number of CPU cores)")
parser.add_argument('--save_model_freq', default=5_000, type=int,
        help="Number of interations until saving the model again \
                (will overwrite last model saved)")

if __name__=='__main__':
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = Model(**args)
    model.train()
