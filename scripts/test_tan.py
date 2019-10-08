import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
from pprint import pformat, pprint
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--sample_std', type=float, default=1.0)
parser.add_argument('--num_vis', type=str, default=10)
parser.add_argument('--num_batches', type=int, default=10)
args = parser.parse_args()
params = HParams(args.cfg_file)
params.num_samples = args.num_samples
params.sample_std = args.sample_std
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
save_dir = os.path.join(params.exp_dir, f'test_{args.sample_std}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(save_dir + '/test.txt', 'w')
f.write(pformat(args))
f.write('\n')
############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)

model = get_model(params)
model.build(trainset, validset, testset)

##########################################################


def batch_psnr(sam, gt):
    '''
    sam: [B,N,H,W,C]
    gt : [B,H,W,C]
    '''
    sam = sam.astype('float32') / 255.
    gt = gt.astype('float32') / 255.
    psnr = []
    for b in range(gt.shape[0]):
        cur_sam = sam[b]
        cur_gt = gt[b]
        cur_psnr = []
        for pred in cur_sam:
            cur_psnr.append(compare_psnr(cur_gt, pred, data_range=1))
        psnr.append(cur_psnr)
    return np.array(psnr)


def batch_show(path, sam, gt, mask, num):
    sam = sam.astype(np.uint8)
    B, N, H, W, C = sam.shape
    num = min(num, B)
    for n in range(num):
        cur_gt = gt[n]
        cur_sam = sam[n]
        cur_mask = mask[n]
        cur_in = cur_gt * cur_mask + 128 * (1 - cur_mask)
        cur_sam = cur_sam.transpose(1, 0, 2, 3).reshape([H, N * W, C])
        img = np.concatenate([cur_in, cur_sam, cur_gt], axis=1)
        if C == 1:
            img = np.squeeze(img, axis=-1)
        plt.imsave(path + f'/{n}.png', img)

    return num


##########################################################
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

weights_dir = os.path.join(params.exp_dir, 'weights')
logging.info(f'Restoring parameters from {weights_dir}')
restore_from = tf.train.latest_checkpoint(weights_dir)
saver.restore(sess, restore_from)

log_likel = []
psnr = []
num_steps = testset.num_steps if args.num_batches < 0 else args.num_batches
testset.initialize(sess)
num_show = 0
for i in range(num_steps):
    ll, gt, mask = sess.run(
        [model.test_ll, testset.x, testset.m])
    sam = model.sample(sess, gt, mask)
    log_likel.append(ll)
    psnr.append(batch_psnr(sam, gt))

    if num_show < args.num_vis:
        num_show += batch_show(save_dir, sam, gt, mask, args.num_vis)

log_likel = np.concatenate(log_likel, axis=0)
psnr = np.concatenate(psnr, axis=0)

f.write(f'log_likel: {log_likel.shape} avg: {np.mean(log_likel)}\n')
f.write(f'psnr: {psnr.shape} avg: {np.mean(psnr)}\n')
