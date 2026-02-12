from gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("""{}. (HINT: you need to install mujoco_py, and also perform the setup 
                                       instructions here: https://github.com/openai/mujoco-py/.)""".format(e))
from gym import utils, spaces
from gym_robotics.envs import rotations
from gym_robotics.envs import utils as gym_robotics_utils
import numpy as np
import os
import xml.etree.ElementTree as ET
import copy
from gym.utils import seeding
from envs.mujoco.mujoco_utils import MujocoTrait
import datetime
import sys

DEFAULT_SIZE = 500
OBJECT_OHE = {'grip': np.array([1, 0, 0, 0, 0]), 'ball': np.array([0, 1, 0, 0, 0]), 'box': np.array([0, 0, 1, 0, 0]), 
              'desk': np.array([0, 0, 0, 1, 0]), 'hammer': np.array([0, 0, 0, 0, 1])}

class MultipleFetchPickAndPlaceEnv(MujocoTrait, utils.EzPickle):
    def __init__(self, render_info = False):
        
        self.colors = ['1 0 0 1', '0 1 0 1', '0 0 1 1', '1 1 0 1', '0 1 1 1', '1 0 1 1']
        self.tints = ['0.25 0 0 1', '0 0.25 0 1', '0 0 0.25 1', '0.25 0.25 0 1', '0 0.25 0.25 1', '0.25 0 0.25 1']
        
        self.gripper_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0
        }
        self.n_substeps = 20
        self.n_actions = 4
        self.gripper_extra_height = 0.2
        self.distance_threshold = 0.1
        self.object_names = ['ball', 'box', 'desk', 'hammer']
        self.object_qty = 4
        self.spec = None

        date_and_time = str(datetime.datetime.now()).replace(' ', '_')
        self.path_to_xmls_folder = '/'.join(sys.argv[0].split('/')[:-1]) + '/env_xmls/' + date_and_time
        os.mkdir(self.path_to_xmls_folder)
        self.render_info = render_info

        self._np_random, self._seed = seeding.np_random(0)
        self.created_object_names = self._initialize_sim()

        self.action_space = spaces.Box(-1., 1., shape = (self.n_actions,), dtype='float32')
        
        obs, _ = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape = obs.shape, dtype = 'float32')

    def _create_multiobject_xml(self):
        prefix_to_folder = os.path.join(os.path.dirname(__file__), 'gripper', 'xmls')
        model_path = os.path.join(prefix_to_folder, 'pick_and_place.xml')
        xml_tree = ET.parse(model_path)
        root = xml_tree.getroot()
        worldbody_idx = None

        for i, child in enumerate(root):
            if 'worldbody' == child.tag:
                worldbody_idx = i
                break
        
        worldbody = root[worldbody_idx]
        
        sampled_colors_idx = self._np_random.choice(len(self.colors), self.object_qty, replace=False)
        sampled_colors = [self.colors[i] for i in sampled_colors_idx]
        sampled_tints = [self.tints[i] for i in sampled_colors_idx]
        sampled_object_names = self._np_random.choice(self.object_names, self.object_qty, replace = True)
        created_object_names = []
        self.created_geom_names = []
        object_postfixes = {'{}'.format(object_name): 0 for object_name in sampled_object_names}
        
        for object_name, object_color, object_tint in zip(sampled_object_names, sampled_colors, sampled_tints):
            # Modify names for conflict resolution in case two same objects present in one scene
            # And change colors of geoms
            cur_object = ET.parse(os.path.join(prefix_to_folder, '{}.xml'.format(object_name))).getroot()
            old_cur_object_name = cur_object.attrib['name']
            new_cur_object_name = old_cur_object_name + str(object_postfixes[old_cur_object_name])
            for child in cur_object:
                if child.tag in ['joint', 'site']:
                    name, *postfix = child.attrib['name'].split(':')
                    child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                if child.tag == 'geom':
                    child.attrib['rgba'] = object_color
                    name, *postfix = child.attrib['name'].split(':')
                    child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                    self.created_geom_names.append(child.attrib['name'])
                if child.tag == 'body':
                    for childs_child in child:
                        if childs_child.tag == 'geom':
                            childs_child.attrib['rgba'] = object_tint
                            name, *postfix = childs_child.attrib['name'].split(':')
                            childs_child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                            self.created_geom_names.append(childs_child.attrib['name'])

            object_postfixes[old_cur_object_name] += 1
            cur_object.attrib['name'] = new_cur_object_name

            created_object_names.append(new_cur_object_name)
            worldbody.append(cur_object)
        
        # Fetch table size; needed for sampling of objects in scene
        table_pos, table_size = None, None
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'table0':
                table_geom = child.find('geom')
                table_pos, table_size = child.attrib['pos'], table_geom.attrib['size']
                table_pos = np.array([float(coord) for coord in table_pos.split(' ')])
                table_size = np.array([float(half_length) for half_length in table_size.split(' ')])
                break

        # Change size of target cylinder accordingly to self.distance_threshold
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'floor0':
                for childs_child in child:
                    if childs_child.attrib['name'] == 'target0':
                        radius, height = childs_child.attrib['size'].split(' ')
                        childs_child.attrib['size'] = str(self.distance_threshold) + ' ' + height

        env_xml_path = os.path.join(self.path_to_xmls_folder, 'env{}.xml'.format(self._seed))
        self.env_xml_path = env_xml_path

        with open(env_xml_path, 'w') as f:
            xml_tree.write(f, encoding='unicode')
        return created_object_names, env_xml_path, table_pos, table_size
    
    def _prepare_object_positions(self, table_pos, table_size):
        # All center of masses must be at least on distance 0.07 from end of table
        # 0.07 - maximal distance from center of masses in created objects
        safe_margin = 0.08

        # All objects must be on distance of 0.15 from each other (in order to not overlap)
        safe_distance = 0.15

        # Ascension above table, in order for all objects to be above it
        safe_ascension = 0.04

        done = False
        while not done:
            done = True
            # Get uniform distribution in [-1., 1.]
            points = (self._np_random.uniform(size = (5, 2)) - 0.5) * 2
            # Convert uniform distribution into distribution with table size, inside safe zone
            points[:, 0] = points[:, 0] * (table_size[0] - safe_margin) + table_pos[0]
            points[:, 1] = points[:, 1] * (table_size[1] - safe_margin) + table_pos[1]
            
            # Check if all objects are on safe distance from eachother
            for i in range(points.shape[0]):
                for j in range(points.shape[0]):
                    if i != j:
                        if np.linalg.norm(points[i] - points[j]) < safe_distance:
                            done = False
        points = np.concatenate([points, np.ones([points.shape[0], 1]) * (table_pos[2] + table_size[2] + safe_ascension)],
                                axis = -1)
        
        return points

    def _initialize_sim(self):
        created_object_names, xml_path, table_pos, table_size = self._create_multiobject_xml()
        model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps = self.n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        initial_qpos = copy.deepcopy(self.gripper_qpos)
        object_positions = self._prepare_object_positions(table_pos = table_pos, table_size = table_size)
        for i, object_name in enumerate(created_object_names):
            initial_qpos['{}:joint'.format(object_name)] = [*object_positions[i], 1., 0., 0., 0.]

        self._env_setup(initial_qpos = initial_qpos)
        return created_object_names
    
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        gym_robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
    
    def _get_obs(self):
        # positions
        grip_desc = get_gripper_description(sim = self.sim)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        # Description of gripper head (on which spatulas are connected)
        grip_pos, grip_rot = grip_desc['pos'], grip_desc['rot']

        # Description of gripper grippers (spatulas, rectangular thing with which gripper grasps object)
        robot_qpos, robot_qvel = gym_robotics_utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        objects_description = {'object_pos': [grip_pos], 'object_rot': [grip_rot],
                               'object_meta': [np.concatenate([gripper_state, gripper_vel], axis = 0)],
                               'object_ohe': [OBJECT_OHE['grip']]}

        for name in self.created_object_names:
            objects_description['object_pos'].insert(0, self.sim.data.get_site_xpos(name))
            # rotations
            objects_description['object_rot'].insert(0, rotations.mat2euler(self.sim.data.get_site_xmat(name)))

            objects_description['object_meta'].insert(0, np.zeros(objects_description['object_meta'][-1].shape))
            
            # Remove number from name, thus excluding last char of name
            objects_description['object_ohe'].insert(0, OBJECT_OHE[name[:-1]])
        
        obs = np.concatenate([objects_description[key] for key in objects_description.keys()], axis = -1)

        info_dict = {
            'coordinates': np.stack(objects_description['object_pos'], axis = 0)[:, :-1].astype(np.float32)
        }
        return obs.astype(np.float32), info_dict

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        gym_robotics_utils.ctrl_set_action(self.sim, action)
        gym_robotics_utils.mocap_set_action(self.sim, action)
    
    def reset(self, seed = None):
        os.remove(self.env_xml_path)
        if seed is not None:
            self._np_random, self._seed = seeding.np_random(seed)
        self.created_object_names = self._initialize_sim()
        obs = self._get_obs()[0]
        return obs
    
    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}
        os.remove(self.env_xml_path)
        os.rmdir(self.path_to_xmls_folder)
    
    def step(self, action):
        prev_obs, prev_info_dict = self._get_obs()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        cur_obs, cur_info_dict = self._get_obs()

        info = {}
        info['before_coordinates'] = np.zeros((len(self.created_object_names) + 1, 2))
        info['after_coordinates'] = np.zeros((len(self.created_object_names) + 1, 2))
        for obj in range(len(self.created_object_names) + 1):
            info['before_coordinates'][obj] = prev_info_dict['coordinates'][obj]
            info['after_coordinates'][obj] = cur_info_dict['coordinates'][obj]
        if self.render_info:
            info['render'] = self.render()
        
        reward, done = 0, False
        return cur_obs, reward, done, info
    
    def render(self):
        width, height = DEFAULT_SIZE, DEFAULT_SIZE
        
        self._get_viewer('rgb_array').render(width, height, segmentation = False)
        # window size used for old mujoco-py:
        data = self._get_viewer('rgb_array').read_pixels(width, height, depth = False, segmentation = False).astype(np.uint8)
        # original image is upside-down, so flip it
        return np.moveaxis(data[::-1, :, :], source = 2, destination = 0)

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self
    
    @property
    def n_obj(self):
        # Include gripper as well
        return self.object_qty + 1
    
    def env_discretizer(self):
        return lambda x: np.floor(x / 0.05)

# ============= Override of MujocoTrait methods =============

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = {}
        for trajectory in trajectories:
            for element in range(trajectory['env_infos']['coordinates'].shape[1] // 3):
                if element not in coordinates_trajectories:
                    coordinates_trajectories[element] = []
                coordinates_trajectories[element].append(\
                    trajectory['env_infos']['coordinates'][:, element * 3: (element * 3 + 2)])
                coordinates_trajectories[element][-1] = np.concatenate([coordinates_trajectories[element][-1], 
                                                                   trajectory['env_infos']['next_coordinates'][:, element * 3: (element * 3 + 2)]],
                                                                   axis = 0)
        return coordinates_trajectories
        

# ============= No change from fetch_env.PickAndPlaceEnv =============

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('table0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -55.
    
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    
def get_gripper_description(sim):
    return {'pos': sim.data.get_site_xpos('robot0:grip'),
            'rot': rotations.mat2euler(sim.data.get_site_xmat('robot0:grip')),
            'velp': sim.data.get_site_xvelp('robot0:grip'),
            'velr': sim.data.get_site_xvelr('robot0:grip')}

def calculate_mean_std(env):
    action = env.action_space.sample()
    