import argparse, functools, sys
import comet_ml
import numpy as np
import omegaconf
import pathlib
import torch
import math

from envs.utils.consistent_normalized_env import consistent_normalize, get_normalizer_preset
from envs.mujoco.obs_wrapper import ExpanderWrapper
from ReplayBuffers.path_replay_buffer import PathBuffer
from RL.skill_model.metra_v2 import METRA
from networks.extractors.static_extractor import get_object_extractor
from RL.policies.sac import SAC
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from eval_utils.traj_utils import draw_2d_gaussians, render_trajectories, calc_eval_metrics
from eval_utils.eval_utils import StatisticsCalculator, monte_carlo_value_difference, calculate_validation_rewards
from eval_utils.video_utils import record_video
import matplotlib.pyplot as plt
import matplotlib

def set_seed(seed):
    import random
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_config():
    parser = argparse.ArgumentParser(prog = 'Metra')
    parser.add_argument('--default_config')
    parser.add_argument('--env_config')
    parser.add_argument('--seed')
    args = parser.parse_args()

    config_folder = str(pathlib.Path(__file__).parent.resolve()) + '/configs'
    rl_config_path = config_folder + '/rl_algos/' + args.default_config
    rl_config = omegaconf.OmegaConf.load(rl_config_path)

    env_config_path = config_folder + '/' + args.env_config
    env_config = omegaconf.OmegaConf.load(env_config_path)
    env_config_name = args.env_config.split('/')[-1][:-5]
    rl_config.merge_with(env_config)
    rl_config.rl_algo.discount = eval(rl_config.rl_algo.discount)
    rl_config.globals.seed = int(args.seed)
    assert rl_config.globals.seed > 3, 'Seeds 0, 1, 2, 3 reserved for evaluation'

    return rl_config, env_config_name

def make_env(env_name, env_kwargs, max_path_length, seed, frame_stack, normalizer_type, 
             render_info = False):
    if env_name == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
        env = ExpanderWrapper(env)
    elif env_name == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw = 100)
        env.seed(seed = seed)
    elif env_name == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw = 100, seed = seed, render_info = render_info)
        env = ExpanderWrapper(env)
        env.seed(seed = seed)
    elif env_name == 'gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names, 
                                           render_info = render_info)
        env = ExpanderWrapper(env)
    elif env_name == 'pixel_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'pixels', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names,
                                           render_info = render_info)
    elif env_name == 'decoupled_gripper':
        from envs.mujoco.gripper_env import MultipleFetchPickAndPlaceEnv
        env = MultipleFetchPickAndPlaceEnv(seed = seed, obs_type = 'decoupled_state', object_qty = env_kwargs.object_qty,
                                           object_names = env_kwargs.object_names,
                                           render_info = render_info)
    elif env_name == 'decoupled_shapes':
        from envs.shapes.push_env.push import PushEnv
        env = PushEnv(seed = seed, arena_size = 8, render_mode = 'state', 
                      render_info = render_info, num_objects_range = env_kwargs.num_objects_range)
    elif env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        if env_name == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif env_name == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert seed is None, 'For some strange reason, this environment does not have any seed...'
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)

    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = False, **normalizer_kwargs)
    elif normalizer_type == 'squashed':
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = True, 
                                   mean = 255. / 2, std = 255. / 2)
    elif normalizer_type == 'preset':
        normalizer_name = env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, flatten_obs = False, normalize_obs = True, mean = normalizer_mean, std = normalizer_std, 
                                   **normalizer_kwargs)
    return env
    
def run():
    config, run_name = fetch_config()

    exp = comet_ml.start(project_name = 'metra')
    exp.log_parameters(config)
    exp.set_name(run_name)

    print('ARGS: ' + str(config))
    if config.globals.n_thread is not None:
        torch.set_num_threads(config.globals.n_thread)

    set_seed(config.globals.seed)
    # TODO check that seeding is correct, meaning that observations must be uncorrelated!
    make_seeded_env = functools.partial(make_env, env_name = config.env.name, 
                                        env_kwargs = config.env.env_kwargs,
                                        max_path_length = config.env.max_path_length,
                                        frame_stack = config.env.frame_stack, 
                                        normalizer_type = config.env.normalizer_type)
    env = make_seeded_env(seed = config.globals.seed, render_info = False)
    
    # TODO change shape
    rl_algo = SAC(obs_length = env.observation_space.shape[1], task_length = config.skill.dim_option, 
                  action_length = env.action_space.shape[0], actor_config = config.rl_algo.policy, obj_qty = env.n_obj,
                  critic_config = config.rl_algo.critics, pooler_config = config.rl_algo.slot_pooler,
                  alpha = config.rl_algo.alpha.value, tau = config.rl_algo.tau,
                  env_spec = env, device = config.globals.device,
                  discount = config.rl_algo.discount, lr = config.rl_algo.lr)

    replay_buffer = PathBuffer(capacity_in_transitions = int(config.replay_buffer.common.max_transitions), 
                               batch_size = config.replay_buffer.common.batch_size, pixel_keys = {}, 
                               seed = config.globals.seed)

    metra = METRA(obs_length = env.observation_space.shape[1], num_objs = env.n_obj, 
                  pooler_config = config.skill.slot_pooler, 
                  traj_encoder_config = config.skill.trajectory_encoder,
                  lr = config.skill.lr,
                  dual_lam = config.skill.dual_lam,
                  option_size = config.skill.dim_option, discrete = config.skill.discrete, 
                  unit_length = config.skill.unit_length, device = config.globals.device,
                  dual_slack = config.skill.dual_slack)
    
    static_object_extractor = get_object_extractor(config = config.static_object_extractor,
                                                   in_dim = env.observation_space.shape[1]).to(config.globals.device)
    env.close()
    
    train_cycle(config.trainer_args, agent = rl_algo, skill_model = metra, static_object_extractor = static_object_extractor, 
                replay_buffer = replay_buffer, make_env_fn = make_seeded_env, seed = config.globals.seed, comet_logger = exp)

def collect_train_trajectories(env, agent, skill_model, 
                               trajectories_length, skills_per_traj = None, n_objects = None, trajectories_qty = None):
    options, obj_idxs = skill_model.sample_options_and_obj_idxs(batch_size = trajectories_qty, 
                                                                traj_len = trajectories_length,
                                                                skills_per_traj = skills_per_traj, 
                                                                n_objects = n_objects)
    return collect_trajectories(env, agent, mode = 'train', options = options, obj_idxs = obj_idxs)

def collect_eval_trajectories(env, agent, options, obj_idxs):
    return collect_trajectories(env, agent, mode = 'eval', 
                                options = options, obj_idxs = obj_idxs)

def collect_trajectories(env, agent, mode, options = None, obj_idxs = None):
    trajectories_qty, trajectories_length = options.shape[:2]
    
    env_qty = len(env.env_fns)
    if isinstance(env, (AsyncVectorEnv, SyncVectorEnv)):
        pseudoepisodes = math.ceil(trajectories_qty / len(env.env_fns))
    else:
        pseudoepisodes = trajectories_qty
    
    cache = []
    for pseudoepisode in range(pseudoepisodes):
        prev_obs, prev_dones = env.reset(), np.full((env_qty,), fill_value = False)
        for i in range(trajectories_length):
            b_opt = options[pseudoepisode * env_qty: (pseudoepisode + 1) * env_qty, i]
            b_obj_idxs = obj_idxs[pseudoepisode * env_qty: (pseudoepisode + 1) * env_qty, i]
            action, action_info = agent.get_actions(prev_obs, b_opt, b_obj_idxs)
            next_obs, rewards, dones, env_infos = env.step(action)
            next_obs = np.transpose(next_obs, [0, 3, 1, 2]) if len(next_obs.shape) == 4 else next_obs
            data = {'observations': prev_obs, 'next_observations': next_obs, 'rewards': rewards, 
                    'dones': dones, 'actions': action, 'options': b_opt, 'obj_idxs': b_obj_idxs}
            if mode == 'train':
                data.update(**action_info)
            elif mode == 'eval':
                env_infos = {key: np.stack([env_infos[i][key] for i in range(len(env_infos))], axis=0) for key in env_infos[0].keys()}
                data.update(**env_infos)

            if pseudoepisode == 0:
                cache.append(data)
            else:
                cache[i] = {key: np.concatenate([cache[i][key], data[key]], axis = 0) for key in cache[i].keys()}
            prev_obs = next_obs
            prev_dones = np.logical_or(prev_dones, dones)
    
    tensors_by_key = {}
    for key in cache[-1].keys():
        tensors_by_key[key] = np.stack([cache[i][key] for i in range(len(cache))], axis = 1)

    return tensors_by_key

def prepare_batch(batch, device = 'cuda'):
    data = {}
    for key, value in batch.items():
        data[key] = torch.from_numpy(value).to(device)
    return data

def train_cycle(trainer_config, agent, skill_model, static_object_extractor, 
                replay_buffer, make_env_fn, seed, comet_logger):
    n_objects = make_env_fn(seed = 0).n_obj
    env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)], context='spawn')
    
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.inference()
        trajs = collect_train_trajectories(env = env, agent = agent, skill_model = skill_model, 
                                           skills_per_traj = trainer_config.skills_per_trajectory, n_objects = n_objects,
                                           trajectories_qty = trainer_config.traj_batch_size, 
                                           trajectories_length = trainer_config.max_path_length)
        prev_cur_step = cur_step
        for traj in trajs['dones']:
            cur_step += len(traj)
        replay_buffer.update_replay_buffer(trajs)
        replay_buffer.prepare_sampling()
        if (replay_buffer.n_transitions_stored < trainer_config.transitions_before_training):
            continue
        
        agent.train()
        object_extractor_stats = StatisticsCalculator('extractor')
        skill_stats = StatisticsCalculator('skill')
        policy_stats = StatisticsCalculator('policy')

        for i in range(trainer_config.trans_optimization_epochs):
            batch = replay_buffer.sample_transitions()
            batch = prepare_batch(batch)
            logs = static_object_extractor.optimize_oe(batch['observations'], batch['next_observations'])
            object_extractor_stats.save_iter(logs)
            extracted_objects = static_object_extractor.extract(batch['observations'])
            logs, rewards = skill_model.train_components(observations = batch['observations'], 
                                                         next_observations = batch['next_observations'],
                                                         static_objects = extracted_objects,
                                                         options = batch['options'],
                                                         obj_idxs = batch['obj_idxs'])
            skill_stats.save_iter(logs)
            logs = agent.optimize_op(observations = batch['observations'], next_observations = batch['next_observations'], 
                                     obj_idxs = batch['obj_idxs'], options = batch['options'], 
                                     actions = batch['actions'], dones = batch['dones'], rewards = rewards)
            policy_stats.save_iter(logs)
        
        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(skill_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(policy_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(object_extractor_stats.pop_statistics(), step = cur_step)
        
        if (prev_cur_step // trainer_config.eval_frequency) < (cur_step // trainer_config.eval_frequency):
            eval_metrics(make_env_fn, agent, skill_model, static_object_extractor,
                         num_random_trajectories = 48, traj_length = trainer_config.max_path_length,
                         gamma = agent.discount, sample_processor = replay_buffer.preprocess_data,
                         device = "cuda:0", comet_logger = comet_logger, step = cur_step)
        prev_cur_step = cur_step

def render_ori_trajectories(options, colors, n_slots, n_objects, eval_env_maker, agent):
    fig, axs = plt.subplots(nrows = n_slots, ncols = n_objects)
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [[axs]]
    
    random_trajectories = []
    for slot_i in range(n_slots):
        obj_idxs = np.zeros(options.shape[:-1], dtype = np.int32) + slot_i
        eval_env = eval_env_maker()
        obj_trajectories = collect_eval_trajectories(env = eval_env, agent = agent, 
                                                     options = options, obj_idxs = obj_idxs)
        eval_env.close()
        del eval_env
        
        for obj_i in range(n_objects):
            coordinates = obj_trajectories['coordinates'][:, :, obj_i]
            last_coordinate = obj_trajectories['next_coordinates'][:, -1:, obj_i]
            # Pad coordinates with last observed position, whilst it is guaranteed that options are consistent
            # across trajectories, thus simply copy last options color one additional time
            tmp_coordinates = np.concatenate([coordinates, last_coordinate], axis = 1)
            axs[slot_i][obj_i].set_title(f'Slot №{slot_i} Object №{obj_i}')
            render_trajectories(tmp_coordinates, colors, None, axs[slot_i][obj_i])
        random_trajectories.append(obj_trajectories)
    fig.canvas.draw()
    skill_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
    skill_img = skill_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    for sub_axs in axs:
        for ax in sub_axs:
            ax.clear()
    plt.close(fig)
    return skill_img, random_trajectories

def render_phi_plot(skill_model, static_object_extractor, 
                    random_trajectories, eval_color, sample_processor, device):
    fig, axs = plt.subplots(1, len(random_trajectories))
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [axs]

    for i, random_trajectories_per_slot in enumerate(random_trajectories):
        data = sample_processor(random_trajectories_per_slot)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(device) for ob in data['observations']]).float()
        last_obj_idx = torch.tensor([ob[-1].item() for ob in data['obj_idxs']]).to(device)
        
        extracted_objects = static_object_extractor.extract(last_obs)
        means, stds, samples = skill_model.fetch_encoder_representation(last_obs, extracted_objects, last_obj_idx)
        axs[i].set_title(f'Object №{i}')
        draw_2d_gaussians(means, stds, eval_color, axs[i])
        draw_2d_gaussians(samples, [[0.03, 0.03]] * len(samples), eval_color, axs[i], fill=True, 
                          use_adaptive_axis=True)
    fig.canvas.draw()
    phi_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    phi_img = phi_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    for ax in axs:
        ax.clear()

    plt.close(fig)
    return phi_img

def eval_metrics(make_env_fn, agent, skill_model, static_object_extractor, 
                 num_random_trajectories, traj_length, gamma, sample_processor, device, comet_logger, step):
    example_env = make_env_fn(seed = 0)
    traj_env_maker = lambda: AsyncVectorEnv([lambda: make_env_fn(seed = i) for i in range(4)], context = 'spawn')
    eval_options, eval_color = skill_model.sample_eval_options(num_random_trajectories, traj_length)
    n_objects = example_env.n_obj
    n_slots = n_objects

    agent.eval()
    skill_img, option_trajectories = render_ori_trajectories(options = eval_options, colors = eval_color, 
                                                             n_slots = n_slots, n_objects = n_objects, eval_env_maker = traj_env_maker, 
                                                             agent = agent)
    comet_logger.log_image(image_data = skill_img, name = "Skill trajs", step = step)
    phi_img = render_phi_plot(skill_model = skill_model, static_object_extractor = static_object_extractor, 
                              random_trajectories = option_trajectories, 
                              eval_color = eval_color, sample_processor = sample_processor, device = device)
    comet_logger.log_image(image_data = phi_img, name = "Phi plot", step = step)
    comet_logger.log_metrics({'Val/gathered_reward': calculate_validation_rewards(trajectories = option_trajectories,
                                                                                  static_object_extractor = static_object_extractor,
                                                                                  skill_model = skill_model)})
    # Videos
    videos = []
    video_options = skill_model.sample_fixated_options(traj_length)
    for slot_idx in range(n_slots):
        video_env = AsyncVectorEnv([lambda: make_env_fn(seed = i, render_info = True) for i in range(2)], context = 'spawn')
        obj_idxs = np.zeros(video_options.shape[:-1], dtype = np.int32) + slot_idx
        video_trajectories = collect_eval_trajectories(env = video_env, agent = agent, options = video_options, obj_idxs = obj_idxs)
        videos.append(video_trajectories['render'])
        video_env.close()

    agent.train()
    for i, skills_videos in enumerate(videos):
        path_to_video = record_video(skills_videos, skip_frames = 2)
        comet_logger.log_video(file = path_to_video, name = f'Slot №{i}'.format(i), step = step)
    
    mets = {}
    if n_objects == 1:
        mets.update(calc_eval_metrics(option_trajectories[0]['coordinates'], example_env.env_discretizer()))
    else:
        for obj_idx in range(n_objects):
            mets.update(calc_eval_metrics(option_trajectories[obj_idx]['coordinates'][:, :, obj_idx], 
                                          example_env.env_discretizer(), 
                                          prefix = f'Object№{obj_idx}'))
    for obj_idx in range(n_objects):
        rewards = skill_model.calculate_rewards(observations = option_trajectories[obj_idx]['observations'], 
                                                next_observations = option_trajectories[obj_idx]['next_observations'],
                                                static_object_extractor = static_object_extractor,
                                                options = option_trajectories[obj_idx]['options'], 
                                                obj_idxs = option_trajectories[obj_idx]['obj_idxs'])
        values = agent.inference_value(observations = option_trajectories[obj_idx]['observations'],
                                       actions = option_trajectories[obj_idx]['actions'],
                                       options = option_trajectories[obj_idx]['options'],
                                       obj_idxs = option_trajectories[obj_idx]['obj_idxs'])

        mc_value_differences = monte_carlo_value_difference(rewards, gamma = gamma)
        predicted_value_differences = values - values[:, -2: -1].repeat(values.shape[1], axis = 1) *\
            (np.fliplr(np.cumprod(np.ones(values.shape) * gamma, axis = 1)) / gamma)
        mets.update({f'Truncated_returns№{obj_idx}': np.mean(mc_value_differences)})
        mets.update({f'Predicted_truncated_returns№{obj_idx}': np.mean(predicted_value_differences)})
        mets.update({f'Mean_error№{obj_idx}': np.mean(mc_value_differences - predicted_value_differences)})
        mets.update({f'Mean_absolute_error№{obj_idx}': np.mean(np.abs(mc_value_differences - predicted_value_differences))})
        mets.update({f'Values_№{obj_idx}': np.mean(values)})
    mets = {f'Object_val/{key}': val for key, val in mets.items()}
    comet_logger.log_metrics(mets, step = step)
    example_env.close()
    del example_env, video_env

if __name__ == '__main__':
    matplotlib.use('Agg')
    run()