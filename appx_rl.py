# from __future__ import print_function

import sys
import numpy as np
import scipy.ndimage as nimg
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import collections
import pydicom as dicom
import os

C_scales = [0.125, 0.25, 0.5, 1.0]
C_scale_id = 1


class Dicomlib:

    def load_dicom(self, PathDicom='test01'):
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(PathDicom):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName, filename))

        # Get ref file
        RefDs = dicom.read_file(lstFilesDCM[0])

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        ConstRescaleSlope = float(RefDs.RescaleSlope)
        ConstRescaleIntercept = float(RefDs.RescaleIntercept)

        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = dicom.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        return ArrayDicom, ConstPixelSpacing, ConstRescaleSlope, ConstRescaleIntercept

    def dicom_to_cropped_vox(self, dicom_path):
        global C_crop_start, C_pad, C_vol
        print('loading dicom series from ', dicom_path, '...')
        vox, ps, rs, ri = self.load_dicom(dicom_path)
        print('performing auto-crop and normalization...')
        hu = vox * rs + ri
        hu[hu < -400] = -400
        hu[hu > 400] = -400
        hu = hu + 400
        hu = hu / 800.0
        SZ = hu.shape

        C_vol = np.uint8(hu*255.0)

        wSize = np.array([350, 350, 350])
        wSize = np.int32(np.floor(wSize / 2.0))

        ctf = np.array([0.76, 0.6, 0.9])

        hu_pad = np.zeros(shape=[SZ[0] + wSize[0] * 2, SZ[1] + wSize[1] * 2, SZ[2] + wSize[2] * 2], dtype=np.float32)
        hu_pad[wSize[0] * 2:wSize[0] * 2 + SZ[0], wSize[1] * 2:wSize[1] * 2 + SZ[1],
        wSize[2] * 2:wSize[2] * 2 + SZ[2]] = hu

        crop_point = np.int32(SZ * ctf + wSize * 2)

        hu_crop = hu_pad[crop_point[0] - wSize[0] * 2:crop_point[0], crop_point[1] - wSize[1] * 2: crop_point[1],
                  crop_point[2] - wSize[2] * 2: crop_point[2]]
        hu_crop_uint8 = np.uint8(hu_crop * 255.0)

        sio.savemat('001.mat', {'hu_crop_uint8': hu_crop_uint8, 'gt': [100, 100, 100]})
        C_crop_start = crop_point - wSize * 2
        C_pad = wSize * 2


class InitPosModel:
    def __init__(self, n_anchors=1, image_dim=44, grid_dim=5, policy_path='net'):
        self.image_dim = image_dim
        self.grid_dim = grid_dim
        self.n_layers = np.int32(np.log2(np.float(self.image_dim) / self.grid_dim))
        self.n_anchors = n_anchors

        self.input = tf.placeholder(shape=[None, self.image_dim, self.image_dim,
                                           self.image_dim, 1], dtype=tf.float32)
        self.gt = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.model_path = policy_path + '/ip/model'
        self.build_model()

    def conv(self, layer_in, k, n, pool=True, activation=None):
        layer = tf.layers.conv3d(inputs=layer_in, kernel_size=[k, k, k], filters=n, activation=activation,
                                 padding='same',
                                 kernel_regularizer=self.regularizer)
        if pool:
            layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[2, 2, 2], strides=[2, 2, 2])
        return layer

    def conv_stack(self, layer):
        self.grid_dim = self.image_dim
        for i in range(self.n_layers):
            layer = self.conv(layer, 3, 16, pool=True, activation=tf.nn.relu)
            self.grid_dim = self.grid_dim // 2
        layer = self.conv(layer, 3, 16, pool=False, activation=tf.nn.relu)
        return layer

    def build_model(self):
        self.output = self.conv_stack(self.input)
        self.output = tf.reshape(self.output, shape=[-1, self.grid_dim ** 3 * 16])
        self.output = tf.layers.dense(self.output, 64, activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        self.output = tf.layers.dense(self.output, 3, activation=tf.nn.sigmoid, kernel_regularizer=self.regularizer)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.output - self.gt), axis=1))
        self.loss = self.loss + tf.losses.get_regularization_losses()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_path)

    def predict(self, input):
        pos = self.sess.run(self.output, {self.input: input / 255.0})
        pos = pos * self.image_dim
        return pos


class World:
    def __init__(self):
        self.ws = 16  # half window size
        self.ws2 = self.ws * 2  # for scale 2
        self.state_size = np.array([self.ws2, self.ws2, self.ws2, 1])

    def set_volume(self, v, gt=None):  # v:shape=[X,Y,Z], gt:[x,y,z]
        self.vol = v
        padded_vol = np.zeros(shape=self.vol.shape + self.state_size[:3] * 2, dtype=np.uint8)
        padded_vol[self.state_size[0]:self.state_size[0] + self.vol.shape[0],
        self.state_size[1]:self.state_size[1] + self.vol.shape[1],
        self.state_size[2]:self.state_size[2] + self.vol.shape[2]] = self.vol
        self.padded_vol = padded_vol
        self.gt = gt

    def set_position(self, pos):  # pos: [x,y,z]
        self.init_pos = pos
        self.pos = self.init_pos.copy()

    def get_state(self):
        x, y, z = self.pos[0], self.pos[1], self.pos[2]
        x += self.state_size[0]
        y += self.state_size[1]
        z += self.state_size[2]
        state_scale_1 = self.padded_vol[x - self.ws:x + self.ws, y - self.ws:y + self.ws, z - self.ws:z + self.ws]
        return state_scale_1[..., np.newaxis]

    def get_reward(self, action, step=2):
        pos = self.pos.copy()
        if action == 0:
            pos[0] += step
        elif action == 1:
            pos[0] -= step
        elif action == 2:
            pos[1] += step
        elif action == 3:
            pos[1] -= step
        elif action == 4:
            pos[2] += step
        else:
            pos[2] -= step

        dist_now = np.sqrt(np.sum((pos - self.gt) ** 2))
        dist_prev = np.sqrt(np.sum((self.pos - self.gt) ** 2))
        # print('dist:', dist_now, dist_prev, self.gt, self. pos, pos)
        reward = -1.0
        if dist_now < dist_prev:
            reward = 1.0
        if dist_now <= np.sqrt(3*(step**2)):
            reward = 2.0

        self.pos = pos
        return reward

    def move(self, action):
        reward = self.get_reward(action)
        if np.any(self.pos < 0) or np.any(self.pos >= self.vol.shape):
            self.pos = self.init_pos
            reward = -10.0
            next_state = self.get_state()
        else:
            next_state = self.get_state()
        return reward, next_state

    def load_volume(self, scale=1):
        loaded_data = sio.loadmat('001.mat')
        scaled_volume = nimg.interpolation.zoom(loaded_data['hu_crop_uint8'], C_scales[scale])
        scaled_volume = np.uint8(scaled_volume)
        self.set_volume(scaled_volume, np.int32(loaded_data['gt'][0] * C_scales[scale] + 0.5))


class Agent:
    def __init__(self, policy_path):
        self.env = World()
        self.policy_path = policy_path
        self.build_model()

    def conv(self, layer_in, k, n, pool=True):
        layer = tf.layers.conv3d(inputs=layer_in, kernel_size=[k, k, k],
                                 filters=n, activation=tf.nn.relu, padding='same')
        if pool:
            layer = tf.layers.max_pooling3d(inputs=layer, pool_size=[2, 2, 2], strides=[2, 2, 2])
        return layer

    def policy_func(self, layer_in, n1, n2):
        layer = tf.layers.dense(inputs=layer_in, units=n1, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=n2, activation=None)
        layer = tf.nn.softmax(layer)
        layer = tf.clip_by_value(layer, 0.1, 0.9)
        return layer

    def build_model(self):
        # input: 32
        tf.reset_default_graph()
        self.input = tf.placeholder(shape=[None, self.env.state_size[0],
                                           self.env.state_size[1], self.env.state_size[2],
                                           self.env.state_size[3]], dtype=tf.float32)

        self.layer = self.conv(self.input, 3, 16)  # dim:16
        self.layer = self.conv(self.layer, 3, 16)  # dim:8
        self.layer = self.conv(self.layer, 3, 32)  # dim:4
        self.layer = self.conv(self.layer, 3, 32)  # dim:2
        self.layer = self.conv(self.layer, 3, 64)  # dim:1
        self.layer = tf.reshape(self.layer, shape=[-1, 64])

        # policy nets

        self.policy_x = self.policy_func(self.layer, 8, 2)
        self.policy_y = self.policy_func(self.layer, 8, 2)
        self.policy_z = self.policy_func(self.layer, 8, 2)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.policy_path)

    def pi_x(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_x, {self.input: state})

    def pi_y(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_y, {self.input: state})

    def pi_z(self, state):
        if len(state.shape) == 4:
            state = state[np.newaxis, ...]
        return self.sess.run(self.policy_z, {self.input: state})

    def pi(self, axis, state):
        if axis == 0:
            return self.pi_x(state)
        elif axis == 1:
            return self.pi_y(state)
        else:
            return self.pi_z(state)


def one_step_explore(world, agent, state, axis, epsilon=0.7):
    policy = np.squeeze(agent.pi(axis, state / 255.0))

    action = np.argmax(policy)

    random_action = np.random.randint(0, 2)
    if np.random.random() > epsilon:
        action = random_action

    reward, next_state = world.move(action + axis * 2)

    return state, action, reward, next_state, policy[action]


def set_init_pos_parameters(scale=0):
    global C_init_pos_center, C_init_pos_radii, C_init_pos_radii_multiplier, C_init_pos_offset
    if scale == 0: # center of the volume
        C_init_pos_center = np.array(world.vol.shape) // 2  # for scale-1, volume
        C_init_pos_radii = 5  # spread of the sample space
        C_init_pos_radii_multiplier = 1  # sparsity of the sample space
    elif scale >= 1:

        C_init_pos_radii_multiplier = 2  # sparsity of the sample space


def episode_explore(world, agent, max_step=150, init_pos=None, epsilon=0.7):
    pt, s_, a_, r_, p_, ax_ = [], [], [], [], [], []

    if init_pos is None:
        #set_init_pos_parameters(C_scale_id)
        disparity = np.random.randint(-C_init_pos_radii, C_init_pos_radii + 1, [3])
        disparity *= C_init_pos_radii_multiplier
        init_pos = C_init_pos_center + disparity
        xx, yy, zz = init_pos[0], init_pos[1], init_pos[2]

    else:
        xx, yy, zz = init_pos[0], init_pos[1], init_pos[2]

    #print('init_pos:', [xx,yy,zz])

    world.set_position(np.array([xx, yy, zz]))
    # start = time.clock()
    state = world.get_state()
    # print('state:', time.clock() - start)
    for step in range(max_step):
        to_break = False

        for ax in range(3):
            # start = time.clock()
            state, action, reward, next_state, policy = one_step_explore(world, agent, state, ax, epsilon)
            pt.append(world.pos)
            s_.append(state)
            a_.append(action)
            r_.append(reward)
            p_.append(policy)
            ax_.append(ax)
            # print('n:', time.clock()-start)
            if reward == -10:
                to_break = True
                break
            state = next_state

        if to_break:
            break

    return step+1, pt, s_, a_, r_, p_, ax_


def explore(world, agent, max_episode=8, max_step=50, init_pos=None, epsilon=0.7):
    step_, pt_, s_, a_, r_, p_, ax_ = [], [], [], [], [], [], []
    for episode in range(max_episode):
        step, pt, s, a, r, p, ax = episode_explore(world, agent, max_step=max_step, epsilon=epsilon, init_pos=init_pos)

        step_.append(step)
        s_.extend(s)
        a_.extend(a)
        r_.extend(r)
        p_.extend(p)
        ax_.extend(ax)

        if episode == 0:
            pt_.append(pt)
        else:
            pt_.append(pt)

    return step_, pt_, s_, a_, r_, p_, ax_


def localize_appendix(max_episode=5, max_step=10, epsilon=0.9, policy_path='net'):
    global C_init_pos_center, C_init_pos_radii, C_init_pos_radii_multiplier
    print('sampling optimal initial position...')
    ipmodel = InitPosModel(image_dim=world.vol.shape[1], policy_path=policy_path)
    init_pos = ipmodel.predict(world.vol[np.newaxis, ..., np.newaxis])
    init_pos = np.int32(init_pos + 0.5)
    init_pos = init_pos[0]
    
    print('performing agent-walk...')
    agent = Agent(policy_path+'/pi1/pi_global_best7')

    C_init_pos_center = init_pos
    C_init_pos_radii = 3
    C_init_pos_radii_multiplier = 2

    for sc in range(2):
        step, pt, s, a, r, p, ax = explore(world, agent, max_episode=max_episode, max_step=max_step, init_pos=None,
                                           epsilon=epsilon)

        long_enough_episode = np.array(step) >= 7
        walk_var_array = np.ones(shape=len(pt), dtype=np.float32) * 9999.0
        mean_pt_array = np.zeros(shape=[len(pt), 3], dtype=np.float32)

        for e in range(len(pt)):
            if long_enough_episode[e]:
                walk_var_array[e] = np.sum(np.var(np.array(pt[e])[(step[e] - 5) * 3:(step[e] - 1) * 3, ], axis=0))
                mean_pt_array[e] = np.mean(np.array(pt[e])[(step[e] - 3) * 3:(step[e] - 1) * 3, ], axis=0)
        walk_var = np.mean(walk_var_array[long_enough_episode])
        mean_pt = np.mean(mean_pt_array[long_enough_episode], axis=0)
        #for i in range(len(mean_pt_array[long_enough_episode])):
        #    print(mean_pt_array[i], walk_var_array[i])
        #print(mean_pt)
        #C_init_pos_center = np.int32(mean_pt * 1.0 + 0.5)
        # next scale

        if sc == 0:
            agent.saver.restore(agent.sess, policy_path + '/pi2/pi_global_best72')
            world.load_volume(sc + 2)
            C_init_pos_center = np.int32(mean_pt*2.0+0.5)

    return mean_pt


def draw_result(pos, pos_rescaled):
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(np.squeeze(world.vol[pos[0], :, :]).T, cmap='gray')
    plt.plot(pos[1], pos[2], '+', color='red')
    plt.xlabel('Y-view')
    plt.subplot(2, 3, 2)
    plt.imshow(np.squeeze(world.vol[:, pos[1], :]).T, cmap='gray')
    plt.plot(pos[0], pos[2], '+', color='red')
    plt.xlabel('X-view')
    plt.subplot(2, 3, 3)
    plt.imshow(np.squeeze(world.vol[:, :, pos[2]]), cmap='gray')
    plt.plot(pos[1], pos[0], '+', color='red')
    plt.xlabel('Z-view')
    world.set_volume(C_vol)
    pos = pos_rescaled.copy()
    plt.subplot(2, 3, 4)
    plt.imshow(np.squeeze(world.vol[pos[0], :, :]).T, cmap='gray')
    plt.plot(pos[1], pos[2], '+', color='red')
    plt.xlabel('Y-view (rescaled)')
    plt.subplot(2, 3, 5)
    plt.imshow(np.squeeze(world.vol[:, pos[1], :]).T, cmap='gray')
    plt.plot(pos[0], pos[2], '+', color='red')
    plt.xlabel('X-view (rescaled)')
    plt.subplot(2, 3, 6)
    plt.imshow(np.squeeze(world.vol[:, :, pos[2]]), cmap='gray')
    plt.plot(pos[1], pos[0], '+', color='red')
    plt.xlabel('Z-view (rescaled)')

    while True:
        plt.pause(0.05)


def fetch_arg(name):
    val = None
    try:
        idx = sys.argv.index(name)
        val = sys.argv[idx+1]
    except ValueError:
        val = None
    return val


if __name__ == '__main__':
    dicom_path = fetch_arg('-dicom_path')
    if dicom_path is None:
        print('-dicom_path not given.')
        sys.exit()

    dicomlib = Dicomlib()
    dicomlib.dicom_to_cropped_vox(dicom_path)

    print('creating world for rl-agent...')
    world = World()
    world.load_volume(C_scale_id)

    max_episode = 5
    max_step = 10
    epsilon = 1.0
    policy_path = 'net'

    arg = fetch_arg('-max_episode')
    if arg is not None: max_episode = int(arg)
    arg = fetch_arg('-max_step')
    if arg is not None: max_step = int(arg)
    arg = fetch_arg('-epsilon')
    if arg is not None: epsilon = float(arg)
    arg = fetch_arg('-policy_path')
    if arg is not None: policy_path = arg

    pos = localize_appendix(max_episode=max_episode,
                            max_step=max_step,
                            epsilon=epsilon,
                            policy_path=policy_path)
    pos_original = np.int32(pos+0.5)
    pos = pos*350.0/world.vol.shape
    pos = C_crop_start + pos - C_pad
    pos = np.int32(pos+0.5)
    print('estimated appendix position:', pos_original, 'rescaled_to_original:', pos)
    draw_result(pos_original, pos)

