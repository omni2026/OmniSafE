
# --- Fix ai2thor Flask-Jinja2 compatibility ---
import jinja2
import markupsafe

# Provide missing symbols that Flask (old) expects
if not hasattr(jinja2, "escape"):
    jinja2.escape = markupsafe.escape
if not hasattr(jinja2, "Markup"):
    jinja2.Markup = markupsafe.Markup
# ------------------------------------------------------
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
import copy
import numpy as np
import torch
import random
import sys
import os
import json
import argparse


from runners.ebalf_runner import EBAlfRunner as Runner
from agents.agent import Agent
from env_monkey_patch.eb_alf import EBAlfEnv_init_patch
EBAlfEnv.__init__ = EBAlfEnv_init_patch


seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--server-num", type=str, default='99')
    parser.add_argument("--split", type=str, default='base')
    parser.add_argument("--save_path", type=str, default='imgs/AW_eval')
    parser.add_argument("--qwen_path", type=str, default='../CKPT')
    parser.add_argument("--data_path", type=str, default="../EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json")
    args = parser.parse_args()
    
    START = args.start
    END = args.end
    DISPLAY = args.server_num
    
    env = EBAlfEnv(eval_set=args.split, data_path=args.data_path,  log_path="", down_sample_ratio=1.0, selected_indexes=[], x_display=DISPLAY, start_idx=START)
    SAVE_PATH = f"{args.save_path}-{args.split}"
    
    agent = Agent(args.qwen_path, "eb-alfred")
    agent.env = env
    for idx_episode in range(START, len(env.dataset)):
        if idx_episode >= END:
            break
        
        data = env.dataset[env._current_episode_num]
        env.reset()
        
        objs = set()
        for skill in env.language_skill_set:
            if skill.startswith("find a"):
                obj = skill.split(" ")[2]
                if '_' not in obj:
                    objs.add(obj + " 1")
                else:
                    objs.add(obj.replace("_", " "))
        print(objs)
        save_path_trial = os.path.join(SAVE_PATH, "episode_%d" % idx_episode)
        os.makedirs(save_path_trial, exist_ok=True)
        agent.reset(save_path_trial, obj_list=list(objs))
        
        runner = Runner(env, agent)

        agent.process_task(None, data["instruction"])
        score = runner.run()
        print(f"\n************RESULT FOR TASK {idx_episode}: {score}\n")
        
    env.close()