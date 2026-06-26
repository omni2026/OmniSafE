import os

import gym
from embodiedbench.envs.eb_alfred.thor_connector import ThorConnector
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, get_global_action_space


ALFRED_REWARD_PATH = os.getenv('ALFRED_REWARD_PATH', 'models/config/rewards.json')
ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'spatial', 
    'visual_appearance', 'long_horizon'
]

def EBAlfEnv_init_patch(self, data_path, log_path, eval_set='base', exp_name='', down_sample_ratio=1.0, selected_indexes=[], detection_box=False, resolution=500, x_display='99', start_idx=0):
    """
    Initialize the AI2THOR environment.
    """
    super(EBAlfEnv, self).__init__()
    self.data_path = data_path # patch: assign data_path
    self.reward_config_path = ALFRED_REWARD_PATH
    self.resolution = resolution
    self.env = ThorConnector(x_display=x_display, player_screen_height=resolution, player_screen_width=resolution) # patch: assign x_display number

    # load dataset
    assert eval_set in ValidEvalSets
    self.eval_set = eval_set
    self.down_sample_ratio = down_sample_ratio
    self.dataset = self._load_dataset(eval_set)
    if len(selected_indexes):
        self.dataset = [self.dataset[i] for i in selected_indexes]
    
    # Episode tracking
    self.number_of_episodes = len(self.dataset)
    self._reset = False
    self._current_episode_num = start_idx # patch: start from a given index of task
    self.selected_indexes = selected_indexes
    self._initial_episode_num = 0
    self._current_step = 0
    self._max_episode_steps = 30
    self._cur_invalid_actions = 0
    self._max_invalid_actions = 10
    self._episode_start_time = 0
    self.episode_log = []
    
    # Task-related attributes
    self.episode_language_instruction = ''
    self.episode_data = None
    # Initialize action space
    self.language_skill_set = None
    self.action_space = None

    # env feedback and image save
    # feedback verbosity, 0: concise, 1: verbose
    self.feedback_verbosity = 0
    self.log_path = log_path # patch: assign log_path

    self.detection = detection_box # add detection in image
    self.name_to_id_dict = None
    self.id_to_name_dict = None
    self.language_skill_set = get_global_action_space()
    self.action_space = gym.spaces.Discrete(len(self.language_skill_set))

