import argparse, functools
import comet_ml
import numpy as np
import omegaconf
import pathlib
import torch

from ReplayBuffers.path_replay_buffer import StateSequence
from envs.utils.consistent_normalized_env import consistent_normalize, get_normalizer_preset
from envs.mujoco.obs_wrapper import ExpanderWrapper
from ReplayBuffers.path_replay_buffer import CyclicBuffer
from RL.skill_model.metra_v2 import METRA
from networks.extractors.static_extractor import get_object_extractor
from RL.policies.sac import SAC
from gym.vector import AsyncVectorEnv
from eval_utils.traj_utils import draw_2d_gaussians, render_trajectories, calc_eval_metrics
from eval_utils.eval_utils import StatisticsCalculator, calculate_validation_rewards
from eval_utils.video_utils import record_video
from gym.wrappers import TimeLimit
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt

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

def make_env(env_name, max_path_length, env_kwargs, seed, frame_stack, normalizer_type, 
             render_info = False):
    if env_name == 'half_cheetah':
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
        env = PushEnv(seed = seed, arena_size = env_kwargs.arena_size, render_mode = 'state', 
                      render_info = render_info, num_objects_range = env_kwargs.num_objects_range)
    elif env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        if env_name == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    # elif env_name == 'kitchen':
    #    sys.path.append('lexa')
    #    from envs.lexa.mykitchen import MyKitchenEnv
    #    assert seed is None, 'For some strange reason, this environment does not have any seed...'
    #    env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)
    env = TimeLimit(env, max_episode_steps = max_path_length)

    normalizer_kwargs = {}

    # TODO Implement readeble normalizer_type, meaning it is different for images and states 
    # and different state environments
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
                                        max_path_length = config.env.max_path_length,
                                        env_kwargs = config.env.env_kwargs,
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

    replay_buffer = CyclicBuffer(total_transitions = config.replay_buffer.common.max_transitions, 
                                 env_qty = config.trainer_args.n_parallel, seed = config.globals.seed,
                                 device = config.globals.device, 
                                 observation_shape = env.observation_space.shape, 
                                 observation_dtype = env.observation_space.dtype,
                                 action_shape = env.action_space.shape, 
                                 action_dtype = env.action_space.dtype,
                                 option_shape = config.skill.dim_option, option_dtype = np.float32,)

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

def continue_random_trajectories_generation(vec_env, agent, skill_model, total_steps,
                                            obs = None, terminated = None, truncated = None, 
                                            options = None, obj_idxs = None):
    steps, env_qty = 0, len(vec_env.env_fns)
    state_seq = StateSequence(num_parallel_envs = env_qty)
    assert (obs is None) == (terminated is None) == (truncated is None) == (options is None) == (obj_idxs is None),\
    'When passing previous state, all elements must either be supplied or not supplied'
    
    if terminated is None:
        options, obj_idxs = [], []
        obs = vec_env.reset()
        terminated, truncated = np.full((env_qty,), fill_value = False), np.full((env_qty,), fill_value = False)
        for i in range(env_qty):
            option, obj_idx = skill_model.sample_option_and_obj_idx()
            options.append(option), obj_idxs.append(obj_idx)
        options, obj_idxs = np.array(options), np.array(obj_idxs)
    
    while steps < total_steps:
        action, action_info = agent.get_actions(obs, options, obj_idxs)
        outp_obs, rewards, dones, env_infos = vec_env.step(action)
        truncated = np.array([env_infos[i].get('TimeLimit.truncated', False) for i in range(env_qty)])
        terminated = np.logical_and(dones, np.logical_not(truncated))

        next_obs = deepcopy(outp_obs)
        for i, done in enumerate(dones):
            if done:
                next_obs[i] = env_infos[i]['terminal_observation']

        state_seq.update(states = obs, next_states = next_obs, actions = action, 
                         options = options, obj_idxs = obj_idxs, 
                         terminated = terminated, truncated = truncated)
        
        for i, done in enumerate(dones):
            if done:
                new_option, new_obj_idx = skill_model.sample_option_and_obj_idx()
                options[i], obj_idxs[i] = new_option, new_obj_idx

        obs = outp_obs
        steps += env_qty

    state_seq.finalize_trajectory_data()
    return state_seq, obs, terminated, truncated, options, obj_idxs


def collect_specific_trajectories(vec_env, agent, options, obj_idxs, colors = None):
    steps, env_qty = 0, len(vec_env.env_fns)
    idx_to_episode = np.arange(env_qty)
    total_quantity_of_trajectories = options.shape[0]
    if colors is None:
        colors = np.zeros(total_quantity_of_trajectories)

    cur_options, cur_obj_idxs, cur_colors = options[:env_qty], obj_idxs[:env_qty], colors[:env_qty]
    episodes = {'option': [[] for i in range(env_qty)], 
                'color': [[] for i in range(env_qty)], 
                'coordinate': [[] for i in range(env_qty)],
                'observation': [[] for i in range(env_qty)],
                'obj_idx': [[] for i in range(env_qty)],
                'action': [[] for i in range(env_qty)],
                'render': [[] for i in range(env_qty)]}

    obs = vec_env.reset()
    while np.min(idx_to_episode) < total_quantity_of_trajectories:
        action, action_info = agent.get_actions(obs, cur_options, cur_obj_idxs)
        outp_obs, rewards, dones, env_infos = vec_env.step(action)
        for i in range(env_qty):
            if idx_to_episode[i] < total_quantity_of_trajectories:
                episodes['option'][idx_to_episode[i]].append(cur_options[i])
                episodes['color'][idx_to_episode[i]].append(cur_colors[i])
                episodes['obj_idx'][idx_to_episode[i]].append(cur_obj_idxs[i])
                episodes['action'][idx_to_episode[i]].append(action[i])
                episodes['coordinate'][idx_to_episode[i]].append(env_infos[i]['before_coordinates'])
                episodes['observation'][idx_to_episode[i]].append(outp_obs[i])
                if 'render' in env_infos[i].keys():
                    episodes['render'][idx_to_episode[i]].append(env_infos[i]['render'])
            if dones[i]:
                episodes['coordinate'][idx_to_episode[i]].append(env_infos[i]['after_coordinates'])
                episodes['observation'][idx_to_episode[i]].append(env_infos[i]['terminal_observation'])
                if 'render' in env_infos[i].keys():
                    episodes['render'][idx_to_episode[i]].append(env_infos[i]['render'])
                idx_to_episode[i] = np.max(idx_to_episode) + 1
                
                if (idx_to_episode[i] < total_quantity_of_trajectories):
                    episodes['option'].append([]), episodes['color'].append([]), episodes['obj_idx'].append([])
                    episodes['coordinate'].append([]), episodes['observation'].append([]), episodes['render'].append([])
                    episodes['action'].append([])

                    cur_options[i], cur_obj_idxs[i] = options[idx_to_episode[i]], obj_idxs[idx_to_episode[i]]
                    cur_colors[i] = colors[idx_to_episode[i]]

    for i in range(len(episodes['option'])):
        for key in episodes.keys():
            episodes[key][i] = np.array(episodes[key][i])
    vec_env.close()
    return episodes
        

def train_cycle(trainer_config, agent, skill_model, static_object_extractor, 
                replay_buffer, make_env_fn, seed, comet_logger):
    env = AsyncVectorEnv([lambda: make_env_fn(seed = (seed + i)) for i in range(trainer_config.n_parallel)], context='spawn')

    obs, terminated, truncated, options, obj_idxs = None, None, None, None, None
    prev_cur_step, cur_step = 0, 0
    for i in range(trainer_config.n_epochs):
        agent.inference()
        trajs, obs, terminated, truncated, options, obj_idxs =\
            continue_random_trajectories_generation(vec_env = env, agent = agent, skill_model = skill_model,
                                                    total_steps = trainer_config.collection_steps, obs = obs,
                                                    terminated = terminated, truncated = truncated, options = options,
                                                    obj_idxs = obj_idxs)
        
        prev_cur_step = cur_step
        cur_step += trainer_config.collection_steps
        replay_buffer.update_buffer(trajs)
        replay_buffer.prepare_for_sampling()
        if (replay_buffer.n_transitions_stored < trainer_config.transitions_before_training):
            continue
        
        agent.train()
        object_extractor_stats = StatisticsCalculator('extractor')
        skill_stats = StatisticsCalculator('skill')
        policy_stats = StatisticsCalculator('policy')

        for i in range(trainer_config.trans_optimization_epochs):
            # TODO fix me, changing hardcoded batch size to a config one
            batch = replay_buffer.sample(256)
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
                                     actions = batch['actions'], 
                                     dones = batch['terminated'], 
                                     rewards = rewards)
            policy_stats.save_iter(logs)
        
        if (prev_cur_step // trainer_config.log_frequency) < (cur_step // trainer_config.log_frequency):
            comet_logger.log_metrics(skill_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(policy_stats.pop_statistics(), step = cur_step)
            comet_logger.log_metrics(object_extractor_stats.pop_statistics(), step = cur_step)
        
        if (prev_cur_step // trainer_config.eval_frequency) < (cur_step // trainer_config.eval_frequency):
            eval_metrics(make_env_fn, agent, skill_model, static_object_extractor,
                         num_random_trajectories = 48, gamma = agent.discount, 
                         device = "cuda:0", comet_logger = comet_logger, step = cur_step)
        prev_cur_step = cur_step

def render_coordinate_trajectories(n_slots, n_objects, trajectories):
    fig, axs = plt.subplots(nrows = n_slots, ncols = n_objects)
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [[axs]]
    
    for slot_i in range(n_slots):
        coordinates_of_objects = trajectories['coordinate'][slot_i]
        color = trajectories['color'][slot_i]
        for obj_i in range(n_objects):
            # Pad coordinates with last observed position, whilst it is guaranteed that options are consistent
            # across trajectories, thus simply copy last options color one additional time
            coordinates = [coordinates_of_objects[episode][:, obj_i] for episode in range(len(coordinates_of_objects))]
            axs[slot_i][obj_i].set_title(f'Slot №{slot_i} Object №{obj_i}')
            render_trajectories(coordinates, color, None, axs[slot_i][obj_i])
    fig.canvas.draw()
    skill_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
    skill_img = skill_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    for sub_axs in axs:
        for ax in sub_axs:
            ax.clear()
    plt.close(fig)
    return skill_img

def render_phi_plot(skill_model, static_object_extractor, 
                    random_trajectories, eval_color, device,
                    n_slots):
    fig, axs = plt.subplots(1, n_slots)
    fig.set_size_inches(15, 15)
    if isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [axs]
    
    for slot_i in range(n_slots):
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(device) for ob in random_trajectories['observation'][slot_i]]).float()
        last_obj_idx = torch.tensor([ob[-1].item() for ob in random_trajectories['obj_idx'][slot_i]]).to(device)
        
        extracted_objects = static_object_extractor.extract(last_obs)
        means, stds, samples = skill_model.fetch_encoder_representation(last_obs, extracted_objects, last_obj_idx)
        axs[slot_i].set_title(f'Object №{slot_i}')
        draw_2d_gaussians(means, stds, eval_color, axs[slot_i])
        draw_2d_gaussians(samples, [[0.03, 0.03]] * len(samples), eval_color, axs[slot_i], fill=True, 
                          use_adaptive_axis=True)
    fig.canvas.draw()
    phi_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    phi_img = phi_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    for ax in axs:
        ax.clear()

    plt.close(fig)
    return phi_img

def eval_metrics(make_env_fn, agent, skill_model, static_object_extractor, 
                 num_random_trajectories, gamma, device, comet_logger, step):
    example_env = make_env_fn(seed = 0)
    traj_env_maker = lambda: AsyncVectorEnv([lambda: make_env_fn(seed = i) for i in range(4)], context = 'spawn')
    eval_options, eval_color = skill_model.sample_eval_options(num_random_trajectories)
    n_objects, n_slots = example_env.n_obj, example_env.n_obj

    agent.eval()
    
    # All trajectories[key] is of shape slot_qty x episode_qty x episode_len x key.shape
    all_trajectories = {}
    eval_slots = [np.full(eval_options.shape[0], fill_value = i) for i in range(n_slots)]
    for slot_i in range(n_slots):
        per_slot_trajectories = collect_specific_trajectories(traj_env_maker(), agent = agent, options = eval_options, 
                                                              obj_idxs = eval_slots[slot_i], colors = eval_color)
        if not all_trajectories:
            for key in per_slot_trajectories:
                all_trajectories[key] = [per_slot_trajectories[key]]
        else:
            for key in per_slot_trajectories:
                all_trajectories[key].append(per_slot_trajectories[key])

    skill_img = render_coordinate_trajectories(n_slots = n_slots, n_objects = example_env.n_obj,
                                               trajectories = all_trajectories)
    comet_logger.log_image(image_data = skill_img, name = "Skill trajs", step = step)
    phi_img = render_phi_plot(skill_model = skill_model, static_object_extractor = static_object_extractor, 
                              random_trajectories = all_trajectories, 
                              eval_color = eval_color, device = device, n_slots = n_slots)
    comet_logger.log_image(image_data = phi_img, name = "Phi plot", step = step)
    val_rewards = calculate_validation_rewards(trajectories = all_trajectories,
                                               static_object_extractor = static_object_extractor,
                                               skill_model = skill_model, n_slots = n_slots)
    comet_logger.log_metrics({f'Val/{key}': val_rewards[key] for key in val_rewards.keys()})

    # Videos
    videos = []
    video_options = skill_model.sample_fixated_options()
    for slot_idx in range(n_slots):
        video_env = AsyncVectorEnv([lambda: make_env_fn(seed = i, render_info = True) for i in range(2)], context = 'spawn')
        obj_idxs = np.zeros(video_options.shape[:-1], dtype = np.int32) + slot_idx
        video_trajectories = collect_specific_trajectories(vec_env = video_env, agent = agent, 
                                                           options = video_options, obj_idxs = obj_idxs)
        videos.append(video_trajectories['render'])

    agent.train()
    for i, skills_videos in enumerate(videos):
        path_to_video = record_video(skills_videos, skip_frames = 2)
        comet_logger.log_video(file = path_to_video, name = f'Slot №{i}'.format(i), step = step)
    
    mets = {}
    if n_objects == 1:
        mets.update(calc_eval_metrics([all_trajectories['coordinate'][slot_i][ep_i][:, slot_i] \
                                       for ep_i in range(len(all_trajectories['coordinate'][slot_i]))], example_env.env_discretizer()))
    else:
        for slot_i in range(n_slots):
            mets.update(calc_eval_metrics([all_trajectories['coordinate'][slot_i][ep_i][:, slot_i] \
                                           for ep_i in range(len(all_trajectories['coordinate'][slot_i]))], 
                                          example_env.env_discretizer(), 
                                          prefix = f'Slot№{slot_i}'))
    """    
    for slot_i in range(n_slots):
        for episode in range(len(all_trajectories['observation'][slot_i])):
            obs = all_trajectories['observation'][slot_i][episode][:-1]
            next_obs = all_trajectories['observation'][slot_i][episode][1:]
        rewards = skill_model.calculate_rewards(observations = obs, 
                                                next_observations = next_obs,
                                                static_object_extractor = static_object_extractor,
                                                options = all_trajectories['option'][slot_i], 
                                                obj_idxs = all_trajectories['obj_idx'][slot_i])
        values = agent.inference_value(observations = obs,
                                       actions = all_trajectories['action'][slot_i],
                                       options = all_trajectories['option'][slot_i],
                                       obj_idxs = all_trajectories['obj_idx'][slot_i])

        mc_value_differences = monte_carlo_value_difference(rewards, gamma = gamma)
        predicted_value_differences = values - values[:, -2: -1].repeat(values.shape[1], axis = 1) *\
            (np.fliplr(np.cumprod(np.ones(values.shape) * gamma, axis = 1)) / gamma)
        mets.update({f'Truncated_returns№{slot_i}': np.mean(mc_value_differences)})
        mets.update({f'Predicted_truncated_returns№{slot_i}': np.mean(predicted_value_differences)})
        mets.update({f'Mean_error№{slot_i}': np.mean(mc_value_differences - predicted_value_differences)})
        mets.update({f'Mean_absolute_error№{slot_i}': np.mean(np.abs(mc_value_differences - predicted_value_differences))})
        mets.update({f'Values_№{slot_i}': np.mean(values)})
    mets = {f'Object_val/{key}': val for key, val in mets.items()}
    """
    comet_logger.log_metrics(mets, step = step)
    example_env.close()
    del example_env, video_env

if __name__ == '__main__':
    matplotlib.use('Agg')
    run()