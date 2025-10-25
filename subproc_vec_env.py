import multiprocessing
from collections import OrderedDict
from typing import Sequence

import gym
import numpy as np

from base_vec_env import VecEnv, CloudpickleWrapper

def make_env(env_id, rank, seed=0):
      """
      Utility function for multiprocessed env.

      :param env_id: (str) the environment ID
      :param num_env: (int) the number of environments you wish to have in subprocesses
      :param seed: (int) the inital seed for RNG
      :param rank: (int) index of the subprocess
      """
      def _init():
          env = gym.make(env_id)
          env.seed(seed + rank)
          return env
      
      return _init

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            #dscho mod
            elif cmd == 'get_agent_spaces':
                remote.send(env.agent_space)
            elif cmd == 'get_goal_spaces':
                remote.send(env.goal_space)
            elif cmd == 'obs2each_state':
                env_state, agent_state = env.obs2each_state(data)
                remote.send((env_state, agent_state))
            elif cmd == 'q_control_type':
                remote.send(env.q_control_type)
            elif cmd == 'act_dim':
                remote.send((env.ur3_act_dim, env.gripper_act_dim))
            elif cmd =='dt':
                remote.send(env.dt)
            elif cmd =='_get_ur3_qpos':
                ur3_qpos = env.env._get_ur3_qpos()
                remote.send(ur3_qpos)
            elif cmd =='get_endeff_pos':
                ee_pos = env.env.get_endeff_pos(data)
                remote.send(ee_pos)
            elif cmd =='_get_my_obs_dict':
                obs_dict = env.env._get_my_obs_dict()
                remote.send(obs_dict)
            elif cmd =='get_mid_reward_done':
                mid_reward, mid_done, mid_info = env.env.get_mid_reward_done(data[0], data[1], data[2], data[3], data[4], data[5])  
                remote.send((mid_reward, mid_done, mid_info))
            elif cmd =='_get_agent_obs':
                agent_obs = env.env._get_agent_obs(data)
                remote.send(agent_obs)
            elif cmd =='inverse_kinematics_ee':
                q_des, iter_taken, err, null_obj = env.env.inverse_kinematics_ee(data[0], data[1], data[2])
                remote.send((q_des, iter_taken, err, null_obj))
            else:
                raise NotImplementedError
        except EOFError:
            break
    


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        self.n_envs = n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)




#dscho mod
class DSSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fn, start_method = None, name = None):
        super(DSSubprocVecEnv, self).__init__(env_fn, start_method)
        self.name = name
        self.remotes[0].send(('get_env_agent_spaces', None))
        env_state_space, agent_state_space = self.remotes[0].recv()
        self.remotes[0].send(('get_goal_spaces', None))
        goal_space = self.remotes[0].recv()
        
        self.env_state_space = env_state_space
        self.agent_state_space = agent_state_space
        self.goal_space = goal_space
    # Note : below codes are not for parallelizing, just util functions
    def obs2each_state(self, observations):
        # assert len(self.remotes)==1, 'this method is not for parallel env'
        self.obs2each_state_async(observations)
        return self.obs2each_state_wait()

    def obs2each_state_async(self, observations):
        # for remote, observation in zip(self.remotes, observations): 
        #     remote.send(('obs2each_state', observation))

        self.remotes[0].send(('obs2each_state', observations)) #[ts, dim]
        self.waiting = True

    def obs2each_state_wait(self):    
        # results = [remote.recv() for remote in self.remotes]
        # env_states, agent_states = zip(*results)
        # return np.stack(env_states), np.stack(agent_states)
        results = [self.remotes[0].recv()]

        self.waiting = False
        env_states, agent_states = zip(*results) #return (~, ) and ~ is  [ts, dim] or [dim, ]
        return env_states[0], agent_states[0]
        

#dscho mod
class UR3SubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fn, start_method = None, name = None):
        super(UR3SubprocVecEnv, self).__init__(env_fn, start_method)
        self.name = name

        self.remotes[0].send(('dt', None))
        dt = self.remotes[0].recv()
        self.dt = dt
        
        self.remotes[0].send(('q_control_type', None))
        q_control_type = self.remotes[0].recv()
        self.q_control_type = q_control_type

        self.remotes[0].send(('act_dim', None))
        ur3_act_dim, gripper_act_dim = self.remotes[0].recv()
        self.ur3_act_dim = ur3_act_dim
        self.gripper_act_dim = gripper_act_dim
        
        self.remotes[0].send(('get_goal_spaces', None))
        goal_space = self.remotes[0].recv()
        self.goal_space = goal_space

        self.remotes[0].send(('get_agent_spaces', None))
        agent_space = self.remotes[0].recv()
        self.agent_space = agent_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            command = {
                'ur3': {'type': self.q_control_type, 'command': action[:2*self.ur3_act_dim]},
                'gripper': {'type': 'forceg', 'command': action[-2*self.gripper_act_dim:]}
            }
            remote.send(('step', command))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def _get_ur3_qpos(self):
        self._get_ur3_qpos_async()
        return self._get_ur3_qpos_wait()

    def _get_ur3_qpos_async(self):
        for remote in self.remotes: 
            remote.send(('_get_ur3_qpos', None))
        self.waiting = True

    def _get_ur3_qpos_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        ur3_qpos = results
        return np.stack(ur3_qpos)

    def inverse_kinematics_ee(self, ee_poss, so3_constraint_ftn, arm):
        self.inverse_kinematics_ee_async(ee_poss, so3_constraint_ftn, arm)
        return self.inverse_kinematics_ee_wait()
    
    #TODO: it can't process different so3 constraint or arm.  only ee_poss
    def inverse_kinematics_ee_async(self, ee_poss, so3_constraint_ftn, arm):
        for remote, ee_pos in zip(self.remotes, ee_poss):
            remote.send(('inverse_kinematics_ee', (ee_pos, so3_constraint_ftn, arm)))
        self.waiting = True

    def inverse_kinematics_ee_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        q_dess, iter_takens, errs, null_objs = zip(*results)
        return np.stack(q_dess), np.stack(iter_takens), np.stack(errs), np.stack(null_objs)
        
    def get_endeff_pos(self, arm):
        self.get_endeff_pos_async(arm)
        return self.get_endeff_pos_wait()

    def get_endeff_pos_async(self, arm):
        for remote in self.remotes: 
            remote.send(('get_endeff_pos', arm))
        self.waiting = True

    def get_endeff_pos_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        endeff_pos = results
        return np.stack(endeff_pos)

    def _get_my_obs_dict(self):
        self._get_my_obs_dict_async()
        return self._get_my_obs_dict_wait()

    def _get_my_obs_dict_async(self):
        for remote in self.remotes:
            remote.send(('_get_my_obs_dict', None))
        self.waiting=True

    def _get_my_obs_dict_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # obs_dicts = zip(*results)
        obs_dicts = results
        
        obs_list, hand_list, second_hand_list, objPos_list, objQuat_list, placingGoal_list, obj_grasp_point_list, second_obj_grasp_point_list, q_des_list, agent_obs_list, second_agent_obs_list = \
            [], [], [], [], [], [], [], [], [], [], []
        for obs_dict in obs_dicts:
            assert isinstance(obs_dict, dict)
            obs_list.append(obs_dict['state_observation'])
            hand_list.append(obs_dict['state_hand'])
            second_hand_list.append(obs_dict['state_second_hand'])
            objPos_list.append(obs_dict['state_obj_pos'])
            objQuat_list.append(obs_dict['state_obj_quat'])
            placingGoal_list.append(obs_dict['state_desired_goal'])
            obj_grasp_point_list.append(obs_dict['state_grasp_point'])
            second_obj_grasp_point_list.append(obs_dict['state_second_grasp_point'])
            q_des_list.append(obs_dict['state_desired_qpos'])
            agent_obs_list.append(obs_dict['state_agent_obs'])
            second_agent_obs_list.append(obs_dict['state_second_agent_obs'])
        obs = np.stack([_ for _ in obs_list], axis =0)
        hand = np.stack([_ for _ in hand_list], axis =0)
        second_hand = np.stack([_ for _ in second_hand_list], axis =0)
        objPos = np.stack([_ for _ in objPos_list], axis =0)
        objQuat = np.stack([_ for _ in objQuat_list], axis =0)
        placingGoal = np.stack([_ for _ in placingGoal_list], axis =0)
        obj_grasp_point = np.stack([_ for _ in obj_grasp_point_list], axis =0)
        second_obj_grasp_point = np.stack([_ for _ in second_obj_grasp_point_list], axis =0)
        q_des = np.stack([_ for _ in q_des_list], axis =0)
        agent_obs = np.stack([_ for _ in agent_obs_list], axis =0)
        second_agent_obs = np.stack([_ for _ in second_agent_obs_list], axis =0)

        return dict(
            state_observation=obs,
            state_hand=hand,
            state_second_hand = second_hand,
            state_obj_pos = objPos,
            state_obj_quat = objQuat,
            state_grasp_point = obj_grasp_point, 
            state_second_grasp_point = second_obj_grasp_point,
            state_desired_goal=placingGoal,
            state_desired_qpos = q_des,
            state_agent_obs = agent_obs,
            state_second_agent_obs = second_agent_obs,\
            )

    # mid_temp_rew2, mid_done2 = env.get_mid_reward_done(obs_dict, action2, 'left')
    def get_mid_reward_done(self, obs_dict, mid_acts, final_goals, final_goal_q_dess, arm, dense_rewards=None):
        if dense_rewards == None:
            dense_rewards = np.array([False]*self.n_envs)
        self.get_mid_reward_done_async(obs_dict, mid_acts, final_goals, final_goal_q_dess, arm, dense_rewards)
        return self.get_mid_reward_done_wait()

    def get_mid_reward_done_async(self, obs_dict, mid_acts, final_goals, final_goal_q_dess, arm, dense_rewards):
        # obs_dict : dictionary consist of [bs, dim]
        # parallel_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
        for remote, (parallel_idx, mid_act), final_goal, final_goal_q_des, dense_reward in zip(self.remotes, enumerate(mid_acts), final_goals, final_goal_q_dess, dense_rewards):
            individual_obs_dict = {}
            for key, value in obs_dict.items():    
                individual_obs_dict[key] = value[parallel_idx]
            remote.send(('get_mid_reward_done', (individual_obs_dict, mid_act, final_goal, final_goal_q_des, arm, dense_reward)))
        self.waiting = True

    def get_mid_reward_done_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        mid_rewards, mid_dones, mid_infos = zip(*results)
        return np.stack(mid_rewards), np.stack(mid_dones), np.stack(mid_infos)


    # mid_next_obs2 = env._get_agent_obs('left')
    def _get_agent_obs(self, arm):
        self._get_agent_obs_async(arm)
        return self._get_agent_obs_wait()

    def _get_agent_obs_async(self, arm):
        for remote in self.remotes:
            remote.send(('_get_agent_obs', arm))
        self.waiting = True

    def _get_agent_obs_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        agent_obs = results
        return np.stack(agent_obs)

