import json
import os
import random
import numpy as np
import argparse
import yaml
import time

import alfworld
from env_monkey_patch.aw import get_env_paths_solvable, env_reset_with_idx, modified_oracle_step
alfworld.agents.environment.alfred_thor_env.AlfredThorEnv.get_env_paths = get_env_paths_solvable
alfworld.agents.environment.alfred_thor_env.AlfredThorEnv.reset = env_reset_with_idx
alfworld.agents.controller.OracleAgent.step = modified_oracle_step

import alfworld.agents
from runners.aw_runner import AWRunner as Runner
from agents.agent import Agent
from tqdm import tqdm

import numpy as np
import torch
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_config():
    config_file = "eval_config.yaml"
    assert os.path.exists(config_file), "Invalid config file"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--split", type=str, default='eval_out_of_distribution')
    parser.add_argument("--save_path", type=str, default='imgs/AW_eval')
    parser.add_argument("--qwen_path", type=str, default='../CKPT')
    args = parser.parse_args()
    
    START = args.start
    END = args.end
    SPLIT = args.split
    LEN = {
        "eval_in_distribution": 140, "eval_out_of_distribution": 134
    }[SPLIT]
    SHUFFLE = False
    
    save_path = args.save_path + "-" + SPLIT

    config = load_config()
    env_type = 'AlfredThorEnv'
    env = get_environment(env_type)(config, train_eval=SPLIT)
    env = env.init_env(batch_size=1)
    if SHUFFLE:
        np.random.shuffle(env.json_file_list)
    agent = Agent(args.qwen_path)

    for i_episode in range(LEN):
        if i_episode >= END:
            break
        if i_episode < START:
            continue
        
        obs, info = env.reset(i=i_episode)
        
        task_instruction = obs[0].split("\n\nYour task is to: ")[1]
        _objs = obs[0].split(", you see")[1].split(".\n\nYour ")[0].split(",")
        objs = []
        for obj in _objs:
            if " and " not in obj:
                assert obj.startswith(" a ")
                objs.append(obj[3:])
            else:
                assert obj.startswith(" and a "), _objs
                objs.append(obj[7:])
        print(objs)
        
        save_path_trial = os.path.join(save_path, "episode_%d" % i_episode)
        os.makedirs(save_path_trial, exist_ok=True)
        agent.reset(save_path_trial, obj_list=objs, )
        
        runner = Runner(env, agent)
        
        agent.process_task(None, task_instruction)

        score = runner.run()
        print(f"\n************RESULT FOR TASK {i_episode}: {score}\n")

        
        