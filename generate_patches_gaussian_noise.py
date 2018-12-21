import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *

# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', # Not in use
                    dest='src_dir',
                    default='./data/TrainGN',
                    help='dir of data')
parser.add_argument('--save_dir',
                     dest='save_dir',
                     default='./data',
                     help='dir of patches')
parser.add_argument('--patch_size',
                    dest='pat_size',
                    type=int,
                    default=40,
                    help='patch size')
parser.add_argument('--stride',
                    dest='stride',
                    type=int,
                    default=10,
                    help='stride')
parser.add_argument('--step',
                    dest='step',
                    type=int,
                    default=0,
                    help='step')
parser.add_argument('--batch_size',
                    dest='bat_size',
                    type=int,
                    default=128,
                    help='batch size')
# check output arguments
parser.add_argument('--from_file', # Not in use
                    dest='from_file',
                    default="./data/img_clean_pats.npy",
                    help='get pic from file')
parser.add_argument('--num_pic',
                    dest='num_pic',
                    type=int,
                    default=10,
                    help='number of pic to pick')
parser.add_argument('--tot', # select train or target data
                    dest='target_train',
                    default='target',
                    help='enter target or train')
args = parser.parse_args()


def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    sigma = 25.0
    count = 0
    src_dir = "./data/TargetGN"
    output_file = "img_noisy_pats"
    noisy_img_dir = "./data/TrainGN"

    target_src_dir = "./data/TargetGN" # TODO modify when using MRI data
    target_output_file = "img_target_pats"
    #filepaths = glob.glob(args.src_dir + '/*.png') # not using src_dir from args anymore
    filepaths = glob.glob(src_dir + '/*.png')
    filepaths.sort() # the filepaths list is not in order and we need it to
                     # be because we need target and train image to be a pair
    if isDebug: # if debugging only use 10 images
        filepaths = filepaths[:10]
    print "number of training data %d" % len(filepaths)

    scales = [1, 0.9, 0.8, 0.7]

    # calculate the number of patches
    for i in xrange(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        print "reading image: %s" % filepaths[i]
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # do not change the original img
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            print "number of training data %d" % (count)
            im_h, im_w = img_s.size
            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES

    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    else:
        numPatches = origin_patch_num
    print "total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size)

    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype="uint8")

    count = 0
    random_mode_vals = []
    # generate patches
    for i in xrange(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')
        print "generating target image: %s" % filepaths[i]
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # print newsize
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))  # extend one dimension

            for j in xrange(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        random_mode = random.randint(0,7)
                        random_mode_vals.append(random_mode)
                        #print "img %s :: scale %d, x_val: %d, y_val: %d, random_mode %d" % (filepaths[i], s, x,y,random_mode)
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :],
                                                                   random_mode)
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, target_output_file), inputs) # target contains clean images
    print "size of inputs tensor = " + str(inputs.shape)

    # initialize input again to make sure we don't keep
    # data from the clean images
    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype="uint8")

    count = 0
    # generate patches for target data
    for i in xrange(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')
        print "generating noisy image: %s" % filepaths[i]
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]),
                       int(img.size[1] * scales[s]))
            # print newsize
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))  # extend one dimension
            img_s = img_s + np.random.normal(scale = sigma, size = img_s.shape)

            # make sure element values in image are between 0-255
            img_s = np.floor(img_s) # remove decimal part
            img_s = np.minimum(img_s, 255) # cap positive values at 255
            img_s = np.maximum(img_s, -255) # cap negative values at -255
            img_s = np.minimum(abs(img_s), 255 + img_s) # map positive values to same value and negative values to complements
            ####
            if s == 0: # save only once
                noisyimg = np.clip(img_s, 0, 255).astype('uint8')
                save_images(os.path.join(noisy_img_dir, 'noisy_%s' % (filepaths[i].split("/")[3])),
                            noisyimg)

            for j in xrange(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        random_mode_n = random_mode_vals.pop(0) # FIFO queue
                        #print "img %s :: scale %d, x_val: %d, y_val: %d, random_mode_n %d" % (filepaths[i], s, x,y,random_mode_n)
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                   random_mode_n)
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, output_file), inputs) # the noisy images
    print "size of target tensor = " + str(inputs.shape)
if __name__ == '__main__':
    generate_patches()
