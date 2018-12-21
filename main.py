import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='Set12', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    with load_data(filepath='./data/img_noisy_pats.npy') as data:
        with load_data(filepath='./data/img_target_pats.npy') as target_data:
            # if there is a small memory, please comment this line and uncomment
            # the line99 in model.py
            # normalize the data to 0-1
            data = data.astype(np.float32) / 255.0
            target_data = target_data.astype(np.float32) / 255.0
            # TODO shuffle data and target data using permutations - two methods could be used
            # 1st:
            rng_state = np.random.get_state()
            np.random.shuffle(data)
            np.random.set_state(rng_state)
            np.random.shuffle(target_data)
            # 2nd:
            #perms = np.random.permutation(a.shape[0])
            #a = a[perms,:,:,:]
            # end shuffle
            eval_files = glob('./data/EvalMRInoisy/*.png')
            eval_files.sort() # to create target image pairs
            eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
            eval_target_files = glob('./data/EvalMRIclean/*.png')
            eval_target_files.sort() # to create target image pairs
            eval_target_data = load_images(eval_target_files)  # list of array of different size, 4-D, pixel value range is 0-255
            denoiser.train(data,
                           target_data,
                           eval_data,
                           eval_target_data,
                           batch_size=args.batch_size,
                           ckpt_dir=args.ckpt_dir,
                           epoch=args.epoch,
                           lr=lr,
                           sample_dir=args.sample_dir)


def denoiser_test(denoiser):
    test_files = glob('./data/TestMRI/*.png')
    test_files.sort()
    test_target_files = glob('./data/TargetMRI/*.png')
    test_target_files.sort()
    print("GPUuuuu\n")
    denoiser.test(test_files,
                  test_target_files,
                  ckpt_dir=args.ckpt_dir,
                  save_dir=args.test_dir)


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
