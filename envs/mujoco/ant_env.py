from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


# pylint: disable=missing-docstring
class AntEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 seed,
                 model_path=None,
                 render_hw = 100,
                 render_info=False,
                 ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'ant.xml'

        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.seed_ = seed
        self.np_rng = np.random.default_rng(seed)

        self.render_hw = render_hw
        self.render_info = render_info

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)

    def step(self, a):
        before_xpos = self.sim.data.qpos.flat[0]
        before_ypos = self.sim.data.qpos.flat[1]
        self.do_simulation(a, self.frame_skip)
        after_xpos = self.sim.data.qpos.flat[0]
        after_ypos = self.sim.data.qpos.flat[1]

        done = self._get_done()

        ob = self._get_obs()
        info = dict(
            before_coordinates = np.array([[before_xpos, before_ypos]]),
            after_coordinates = np.array([[after_xpos, after_ypos]])
            )
        if self.render_info:
            info['render'] = self.render(mode='rgb_array', width = self.render_hw, 
                                         height = self.render_hw).transpose(2, 0, 1)

        return ob, 0, done, info

    def _get_obs(self):
        # No crfc observation
        obs = np.concatenate([
            self.sim.data.qpos.flat[:15],
            self.sim.data.qvel.flat[:14],
        ])
        return obs

    def _get_done(self):
        return False

    def reset_model(self):
        qpos = self.init_qpos + self.np_rng.uniform(
                size=self.sim.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_rng.standard_normal(self.sim.model.nv) * .1

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 2.5
        pass
    
    @property
    def n_obj(self):
        # Only ant is an object
        return 1
    
    @property
    def body_com_indices(self):
        return self._body_com_indices

    @property
    def body_comvel_indices(self):
        return self._body_comvel_indices
    
    def env_discretizer(self):
        return np.floor