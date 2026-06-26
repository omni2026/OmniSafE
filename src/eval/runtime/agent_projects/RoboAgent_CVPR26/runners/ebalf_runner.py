import sys

from agents.agent import Agent
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

class EBAlfRunner():
    def __init__(self, env: EBAlfEnv, agent: Agent):
        self.env = env
        self.agent = agent
        self.env_step_id = 0
        
        self.last_recep = None
        self.last_info = None

    def run(self, ):
        steps = 0
        done = False
        while not done:
            exec_actions, actions = self.agent.get_qwen_action()
            
            for exec_action, action in zip(exec_actions, actions):
                sys.stdout.flush()
                if exec_action == "fail":
                    if steps == 0:
                        gcr = 0
                        break
                    gcr = self.last_info['task_success']
                    break
                
                if exec_action != "pass":
                    print(f"--------------------[ENV STEP {steps}]: {exec_action}\n")
                    steps += 1
                done, info = self.execute(exec_action, action)
                if done:
                    gcr = info['task_success']
                    break
                if info['last_action_success'] == 0 and exec_action.split(" ")[0] not in ["open", "close"]: # to save evaluation time, we may give up when the previous action failed, but the failure of open/close is normally not that severe
                    break
            if done or exec_action == "fail":
                break
        return gcr
       
    
    def env_step(self, action, process_obs=True, img_only=False):
        assert action in self.env.language_skill_set, [action, self.env.language_skill_set]
        action_idx = self.env.language_skill_set.index(action)
        obs, reward, done, info = self.env.step(action_idx)
        if process_obs:
            self.agent.process_observation(
                self.env.env.last_event.frame,
                self.env_step_id,
            )
            
        self.env_step_id += 1
        return obs, reward, done, info
        
    def execute(self, action_str, raw_action):
        if action_str.split(" ")[0] in ["pass", "examine"]:
            action_str = "do nothing"
            _ = [""]
            done = False
            info = {'task_success': 0., 'last_action_success': 1}
        else:
            if action_str.startswith("open the ") and action_str not in self.env.language_skill_set:
                done = False
                info = {'task_success': 0., 'last_action_success': 0}
            else:
                try:
                    obs, reward, done, info = self.env_step(action_str)
                except:
                    # print("INVALID ACTION!!!!!!", action_str)
                    if self.last_info:
                        return True, self.last_info
                    return True, {'task_success': 0., 'last_action_success': 0.} 
            
            self.agent.process_feedback(info['last_action_success'] > 0, raw_action)
            self.last_info = info
            
        return done, info
        