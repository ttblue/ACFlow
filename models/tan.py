import tensorflow as tf
import numpy as np

from .rnvp.modules import encoder_spec, decoder_spec
from .rnvp.utils import standard_normal_ll, standard_normal_sample
from .rnvp.logits import preprocess, postprocess
from .pixelcnn.pixelcnn import pixelcnn_spec
from .pixelcnn.mixture import mixture_likelihoods, sample_mixture


class Model(object):
    def __init__(self, hps):
        self.hps = hps

        self.pixelcnn = tf.make_template('PixelCNN', pixelcnn_spec)

    def forward(self, x, m, train, init=False):
        reverse = False
        m = tf.cast(m, tf.float32)
        x, logdet = preprocess(x, self.hps.data_constraint)
        z, ldet = encoder_spec(x, m, self.hps, self.hps.n_scale,
                               use_batch_norm=False, train=train)
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet

        dropout = self.hps.dropout_rate if train else 0.0
        inp = tf.concat([z, m], axis=-1)
        likel_param = self.pixelcnn(
            inp, h=None, hparams=self.hps,
            init=init, dropout_p=dropout)
        prior_ll = mixture_likelihoods(
            likel_param, z, self.hps.image_shape[-1])
        prior_ll = tf.reduce_sum(prior_ll * (1. - m), [1, 2, 3])

        log_likel = prior_ll + logdet

        return log_likel

    def sample_z(self, x, z, m):
        m = tf.cast(m, tf.float32)
        x, _ = preprocess(x, self.hps.data_constraint)
        z = z * (1. - m) + x * m
        inp = tf.concat([z, m], axis=-1)
        likel_param = self.pixelcnn(
            inp, h=None, hparams=self.hps,
            init=False, dropout_p=0.)
        z_sam = sample_mixture(likel_param, self.hps.image_shape[-1])
        z_sam = z_sam * (1. - m) + x * m

        return z_sam

    def sample_x(self, x, z, m):
        reverse = True
        m = tf.cast(m, tf.float32)
        x, _ = preprocess(x, self.hps.data_constraint)
        z = z * (1. - m) + x * m
        x, _ = decoder_spec(z, m, self.hps, self.hps.n_scale,
                            use_batch_norm=False, train=False)
        x, _ = postprocess(x, self.hps.data_constraint)

        return x

    def build(self, trainset, validset, testset):
        train_x, train_m = trainset.x, trainset.m
        valid_x, valid_m = validset.x, validset.m
        test_x, test_m = testset.x, testset.m
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            init_shape = [self.hps.batch_size *
                          self.hps.init_batches] + self.hps.image_shape
            self.init_x = tf.placeholder(tf.uint8, init_shape)
            self.init_m = tf.placeholder(tf.uint8, init_shape)
            self.init = self.forward(self.init_x, self.init_m, True, True)

            self.train_ll = self.forward(train_x, train_m, True)
            self.valid_ll = self.forward(valid_x, valid_m, False)
            self.test_ll = self.forward(test_x, test_m, False)

            _shape = [self.hps.batch_size] + self.hps.image_shape
            self.x_ph = tf.placeholder(tf.uint8, _shape)
            self.z_ph = tf.placeholder(tf.float32, _shape)
            self.m_ph = tf.placeholder(tf.uint8, _shape)
            self.z_sam = self.sample_z(self.x_ph, self.z_ph, self.m_ph)
            self.x_sam = self.sample_x(self.x_ph, self.z_ph, self.m_ph)

            nll = tf.reduce_mean(-self.train_ll)
            tf.summary.scalar('nll', nll)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            loss = nll + self.hps.lambda_reg * l2_reg
            tf.summary.scalar('loss', loss)

            self.global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.inverse_time_decay(
                self.hps.lr, self.global_step,
                self.hps.decay_steps, self.hps.decay_rate,
                staircase=True)
            tf.summary.scalar('lr', learning_rate)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name="Adam")
            grads_and_vars = optimizer.compute_gradients(
                loss, tf.trainable_variables())
            grads, vars_ = zip(*grads_and_vars)
            if self.hps.clip_gradient > 0:
                grads, gradient_norm = tf.clip_by_global_norm(
                    grads, clip_norm=self.hps.clip_gradient)
                gradient_norm = tf.check_numerics(
                    gradient_norm, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gradient_norm', gradient_norm)
            capped_grads_and_vars = zip(grads, vars_)
            self.train_op = optimizer.apply_gradients(
                capped_grads_and_vars, global_step=self.global_step)

            # summary
            self.summ_op = tf.summary.merge_all()

    def sample_once(self, sess, x_in, m_in):
        B, H, W, C = _shape = [self.hps.batch_size] + self.hps.image_shape
        z_nda = np.zeros(_shape, dtype=np.float32)
        m_nda = m_in.copy()
        x_nda = x_in.copy()
        for yi in range(H):
            for xi in range(W):
                feed_dict = {self.x_ph: x_nda,
                             self.z_ph: z_nda,
                             self.m_ph: m_nda}
                new_z = sess.run(self.z_sam, feed_dict)
                z_nda[:, yi, xi, :] = new_z[:, yi, xi, :]
        feed_dict = {self.x_ph: x_nda, self.z_ph: z_nda, self.m_ph: m_nda}
        x = sess.run(self.x_sam, feed_dict)
        x = x.astype(np.uint8)

        return x

    def sample(self, sess, x_in, m_in):
        sams = []
        for i in range(self.hps.num_samples):
            sam = self.sample_once(sess, x_in, m_in)
            sams.append(sam)
        sams = np.stack(sams, axis=1)

        return sams
