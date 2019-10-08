import tensorflow as tf
import numpy as np
image_grid = tf.contrib.gan.eval.image_grid

from .rnvp.modules import encoder_spec, decoder_spec
from .rnvp.utils import standard_normal_ll, standard_normal_sample
from .rnvp.logits import preprocess, postprocess
from .gan_utils import *


def rearrange(image_tensor):
    B, H, W, C = image_tensor.get_shape().as_list()
    L = int(np.sqrt(B))
    img = image_grid(image_tensor[:L * L], [L, L], [H, W], C)

    return img


def discriminator(image, hps):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        if hps.dataset == 'celeba':
            h0 = Layernorm('ln1', [1, 2, 3], lrelu(
                conv2d(image, 64, name='cv1')))
            h1 = Layernorm('ln2', [1, 2, 3], lrelu(
                conv2d(h0, 128, name='cv2')))
            h2 = Layernorm('ln3', [1, 2, 3], lrelu(
                conv2d(h1, 256, name='cv3')))
            h3 = Layernorm('ln4', [1, 2, 3], lrelu(
                conv2d(h2, 256, name='cv4')))
            logits = linear(tf.layers.flatten(h3), 1, 'lg')

    return logits


class Model(object):
    def __init__(self, hps):
        self.hps = hps

    def forward(self, x, m, train):
        '''
        Args:
            x: data, [B,H,W,C] [uint8]
            m: mask, [B,H,W,C] [uint8]
        '''
        reverse = False
        m = tf.cast(m, tf.float32)
        x, logdet = preprocess(x, self.hps.data_constraint)
        z, ldet = encoder_spec(x, m, self.hps, self.hps.n_scale,
                               use_batch_norm=self.hps.use_batch_norm,
                               train=train)
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet
        prior_ll = standard_normal_ll(z)
        prior_ll = tf.reduce_sum(prior_ll * (1. - m), [1, 2, 3])
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, m, train):
        reverse = True
        m = tf.cast(m, tf.float32)
        m = tf.expand_dims(m, axis=1)
        m = tf.tile(m, [1, self.hps.num_samples, 1, 1, 1])
        z = standard_normal_sample(
            [self.hps.batch_size, self.hps.num_samples] + self.hps.image_shape)
        z = z * self.hps.sample_std
        x, _ = preprocess(x, self.hps.data_constraint)
        x = tf.expand_dims(x, axis=1)
        x = z * (1. - m) + x * m
        x = tf.reshape(
            x, [self.hps.batch_size * self.hps.num_samples] + self.hps.image_shape)
        m = tf.reshape(
            m, [self.hps.batch_size * self.hps.num_samples] + self.hps.image_shape)
        x, _ = decoder_spec(x, m, self.hps, self.hps.n_scale,
                            use_batch_norm=self.hps.use_batch_norm,
                            train=train)
        x, _ = postprocess(x, self.hps.data_constraint)
        x = tf.reshape(
            x, [self.hps.batch_size, self.hps.num_samples] + self.hps.image_shape)
        return x

    def inverse_zero(self, x, m, train):
        reverse = True
        m = tf.cast(m, tf.float32)
        z = tf.zeros([self.hps.batch_size] +
                     self.hps.image_shape, dtype=tf.float32)
        x, _ = preprocess(x, self.hps.data_constraint)
        x = z * (1. - m) + x * m
        x, _ = decoder_spec(x, m, self.hps, self.hps.n_scale,
                            use_batch_norm=self.hps.use_batch_norm,
                            train=train)
        x, _ = postprocess(x, self.hps.data_constraint)
        return x

    def build(self, trainset, validset, testset):
        train_x, train_m = trainset.x, trainset.m
        valid_x, valid_m = validset.x, validset.m
        test_x, test_m = testset.x, testset.m
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.train_ll = self.forward(train_x, train_m, True)
            self.valid_ll = self.forward(valid_x, valid_m, False)
            self.test_ll = self.forward(test_x, test_m, False)
            self.train_sam = self.inverse(train_x, train_m, False)
            self.valid_sam = self.inverse(valid_x, valid_m, False)
            self.test_sam = self.inverse(test_x, test_m, False)
            self.train_sam_mean = self.inverse_zero(train_x, train_m, False)
            self.valid_sam_mean = self.inverse_zero(valid_x, valid_m, False)
            self.test_sam_mean = self.inverse_zero(test_x, test_m, False)
            # image summ
            gray_img = tf.ones_like(train_x) * 128
            train_gt = rearrange(train_x)
            tf.summary.image('train/gt', train_gt)
            train_in = rearrange(train_x * train_m + gray_img * (1 - train_m))
            tf.summary.image('train/in', train_in)
            for i in range(self.hps.num_samples):
                train_out = rearrange(self.train_sam[:, i])
                tf.summary.image(f'train/out_{i}', train_out)
            # likelihood loss
            nll = tf.reduce_mean(-self.train_ll)
            tf.summary.scalar('nll', nll)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            loss = nll + self.hps.lambda_reg * l2_reg
            tf.summary.scalar('likel_loss', loss)
            # gan loss V1
            # B, N, H, W, C = self.train_sam.get_shape().as_list()
            # real_image = tf.cast(train_x, tf.float32)
            # real_image = tf.reshape(real_image, [B, H, W, 1, C])
            # real_label = tf.ones([B, 1])
            # fake_image = tf.transpose(self.train_sam, [0, 2, 3, 1, 4])
            # fake_label = tf.zeros([B, N])
            # image = tf.concat([real_image, fake_image], axis=3)
            # label = tf.concat([real_label, fake_label], axis=1)
            # ind = tf.random_shuffle(tf.range(N + 1))
            # image = tf.transpose(image, [3, 0, 1, 2, 4])
            # label = tf.transpose(label, [1, 0])
            # image = tf.batch_gather(image, ind)
            # label = tf.batch_gather(label, ind)
            # image = tf.transpose(image, [1, 2, 3, 0, 4])
            # image = tf.reshape(image, [B, H, W, C * N + C])
            # label = tf.transpose(label, [1, 0])

            # gan loss V2
            B, N, H, W, C = self.train_sam.get_shape().as_list()
            real_x = tf.cast(train_x, tf.float32)
            real_m = tf.cast(train_m, tf.float32)
            real = tf.concat([real_x, real_m], axis=3)
            fake_x = self.train_sam[:, 0]
            fake = tf.concat([fake_x, real_m], axis=3)

            real_logits = discriminator(real, self.hps)
            fake_logits = discriminator(fake, self.hps)

            g_loss = -tf.reduce_mean(fake_logits)
            tf.summary.scalar('g_gan', g_loss)
            d_loss = -tf.reduce_mean(real_logits) + tf.reduce_mean(fake_logits)
            tf.summary.scalar('d_gan', d_loss)
            alpha = tf.random_uniform([B, 1, 1, 1])
            interpolated = real_x + alpha * (fake_x - real_x)
            interpolated = tf.concat([interpolated, real_m], axis=3)
            inter_logits = discriminator(interpolated, self.hps)
            gradients = tf.gradients(inter_logits, [interpolated])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
            d_loss += 10. * gradient_penalty
            g_loss = loss + self.hps.lambda_gan * g_loss
            tf.summary.scalar('g_loss', g_loss)
            tf.summary.scalar('d_loss', d_loss)

            # train
            self.global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.inverse_time_decay(
                self.hps.lr, self.global_step,
                self.hps.decay_steps, self.hps.decay_rate,
                staircase=True)
            tf.summary.scalar('lr', learning_rate)
            g_optim = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name="Adam")
            d_optim = tf.train.AdamOptimizer(
                learning_rate=learning_rate * 3,
                beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name="Adam")

            t_vars = tf.trainable_variables()
            g_vars = [var for var in t_vars if 'discriminator' not in var.name]
            d_vars = [var for var in t_vars if 'discriminator' in var.name]

            d_grad = d_optim.compute_gradients(d_loss, d_vars)
            d_grad_norm = tf.global_norm(d_grad)
            tf.summary.scalar('d_grad_norm', d_grad_norm)
            d_step = d_optim.apply_gradients(d_grad)

            g_grad = g_optim.compute_gradients(g_loss, g_vars)
            g_grad_norm = tf.global_norm(g_grad)
            tf.summary.scalar('g_grad_norm', g_grad_norm)
            g_step = g_optim.apply_gradients(
                g_grad, global_step=self.global_step)

            self.train_op = tf.group([d_step, g_step])

            # summary
            self.summ_op = tf.summary.merge_all()
