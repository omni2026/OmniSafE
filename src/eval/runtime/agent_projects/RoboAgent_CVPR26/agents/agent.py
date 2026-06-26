import torch
import cv2

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from agents.qwen import inference as qwen_inference
from peft import PeftModel


class Agent(object):
    def __init__(self, vlm_model_path, env_name="alfworld"):
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.vlm_processer = AutoProcessor.from_pretrained(vlm_model_path)
        
        self.env_name = env_name
        if env_name == "alfworld":
            from agents.prompt_aw import prompt_og, prompt_sd, prompt_lpe, prompt_lpm, prompt_eg, prompt_es, prompt_ct
            self.prompt_og = prompt_og
            self.prompt_sd = prompt_sd
            self.prompt_lpe = prompt_lpe
            self.prompt_lpm = prompt_lpm
            self.prompt_eg = prompt_eg
            self.prompt_es = prompt_es
            self.prompt_ct = prompt_ct
        elif env_name == "eb-alfred":
            from agents.prompt_ebalf import prompt_og, prompt_sd, prompt_lpe, prompt_lpm, prompt_eg, prompt_es, prompt_ct
            self.prompt_og = prompt_og
            self.prompt_sd = prompt_sd
            self.prompt_lpe = prompt_lpe
            self.prompt_lpm = prompt_lpm
            self.prompt_eg = prompt_eg
            self.prompt_es = prompt_es
            self.prompt_ct = prompt_ct
        else:
            raise ValueError(f"Invalid environment name: {env_name}")
    
    def reset(self, save_path, obj_list):
        self.last_goto = None
        self.save_path = save_path
        self.save_i = 0
        
        self.observed_objects_list = [x for x in sorted(obj_list)]
        self.core_history = ""
        self.explored = []
        self.invent = "nothing"
        self.last_local_traj = []
        self.ability_buffer = []
        self.ability_buffer_idx = 0
        self.last_to_find = None
        self.slice_idx = 3 # for assigning ID of the sliced object in alfworld
        
        with open(f"{self.save_path}/qwen_log.txt", "w") as f:
            f.write("BEGIN!!!\n")
        
    def process_observation(self, rgb, env_step_id):
        cv2.imwrite(f"{self.save_path}/step_{env_step_id}.png", rgb[:, :, ::-1])
        self.cur_rgb_path = f"{self.save_path}/step_{env_step_id}.png"
        
        
    def process_task(self, task_info, task_instruction):
        print("[TASK] ", task_instruction)
        self.task_instruction = task_instruction
        return
    
    def process_feedback(self, message, last_action):
        assert last_action not in ["examine", "pass", "do nothing"]
        if self.env_name == "alfworld":
            assert last_action.split(" ")[0] in ["take", "open", "close", "put", "slice", "heat", "cool", "clean", "go", "use"], last_action
        elif self.env_name == "eb-alfred":
            assert last_action.split(" ")[0] in ["pick", "open", "close", "put", "slice", "turn", "find"], last_action

        self.last_action = last_action
        aid = len(self.last_local_traj) // 2 + 1
        self.last_local_traj.append(f"[action {aid}] {last_action}")
        self.last_local_traj.append(f"[feedback {aid}] {'success' if message else 'failure'}")
        
        self.scene_description = ""
        if message:
            if self.env_name == "alfworld":
                if last_action.startswith("take "):
                    self.invent = last_action.split("take ")[1].split(" from")[0]
                elif last_action.startswith("put "):
                    self.invent = "nothing"
                    if message:
                        self.slice_idx += 1
                elif last_action.startswith("go to"):
                    self.last_goto = last_action.split("go to ")[1]
            elif self.env_name == "eb-alfred":
                if last_action.startswith("pick "):
                    self.invent = last_action.split("pick up the ")[1]
                    if "_" in self.invent:
                        self.invent = self.invent.split("_")[0]
                elif last_action.startswith("put down "):
                    self.invent = "nothing"
                elif last_action.startswith("find a "):
                    self.last_goto = last_action.split("find a ")[1]
                    if "_" in self.last_goto:
                        self.last_goto = self.last_goto.split("_")[0]
        return
    
    def get_qwen_action(self, ):
        raw_actions = self.get_qwen_action_raw()
        
        exec_actions = [] # to be executed in the environment
        actions = [] # to be passes to process_feedback

        if self.env_name == "alfworld":
            for raw_action in raw_actions:
                for x in ["LettuceSliced", "AppleSliced", "PotatoSliced", "TomatoSliced", "BreadSliced"]:
                    assert not f"{x} {None}" in raw_action
                    if f"{x}" in raw_action:
                        raw_action = raw_action.replace(f"{x}", f"sliced-{x[:-6]} {self.slice_idx}")
                exec_actions.append(raw_action)
            actions = exec_actions
        elif self.env_name == "alfworld_text":
            for raw_action in raw_actions:
                for x in ["LettuceSliced", "AppleSliced", "PotatoSliced", "TomatoSliced", "BreadSliced"]:
                    assert not f"{x} {None}" in raw_action
                    if f"{x}" in raw_action:
                        raw_action = raw_action.replace(f"{x}", f"sliced-{x[:-6]} {self.slice_idx}")
                if raw_action.split(" ")[0] == "put":
                    raw_action = raw_action.replace("put ", "move ")
                exec_actions.append(raw_action)
            actions = exec_actions
        elif self.env_name == "eb-alfred":
            for raw_action in raw_actions:
                # rename some objects to match the skill set of EmbodiedBench
                # generally this can be done by computing semantic similarity between the raw_action and the vocabulary of the environment, here we just use a rule-based matching for simplicity
                raw_action = raw_action.replace("Spray bottle", "SprayBottle").replace("the sponge", "the DishSponge").replace(" Sliced ", " ").replace("Floor lamp", "FloorLamp").replace("Desk lamp", "DeskLamp").replace("lettuce", "Lettuce").replace("apple", "Apple").replace("potato", "Potato").replace("tomato", "Tomato").replace("bread", "Bread").replace("the Sponge", "the DishSponge").replace("the Key ", "the KeyChain ").replace("Garbage can", "GarbageCan").replace("the towel", "the HandTowel").replace("Inkpen", "Pen").replace("Watering can", "WateringCan").replace("Armchair", "ArmChair").replace(" sliced ", " ").replace("Glass bottle", "Glassbottle").replace("Soap bottle", "SoapBottle").replace("SlicedApple", "Apple").replace("SlicedLettuce", "Lettuce").replace("SlicedBread", "Bread").replace("SlicedTomato", "Tomato").replace("SlicedPotato", "Potato").replace("GlassBottle", "Glassbottle").replace("the knife", "the Knife").replace("AppleSliced", "Apple").replace("LettuceSliced", "Lettuce").replace("BreadSliced", "Bread").replace("TomatoSliced", "Tomato").replace("PotatoSliced", "Potato")
                if raw_action.endswith("the Key"):
                    raw_action = raw_action.replace("the Key", "the KeyChain")

                # deal with the object index in EB's skill name
                if raw_action.split(" ")[0] in ["find", "open", "close"]:
                    if raw_action.endswith(" 1"):
                        action_str = raw_action[:-2]
                    else:
                        action_str = " ".join(raw_action.split(" ")[:-1]) + "_" + raw_action.split(" ")[-1]
                elif raw_action.split(" ")[0] in ["turn", "slice", "pick"]:
                    if raw_action in ["pick up the Apple", "pick up the Lettuce", "pick up the Tomato", "pick up the Potato", "pick up the Bread"]:
                        action_str = raw_action
                    else:
                        action_str = " ".join(raw_action.split(" ")[:-1])
                elif raw_action.split(" ")[0] in ["put", "pass", "examine", "fail"]:
                    action_str = raw_action
                else:
                    # print(raw_action)
                    # raise NotImplementedError
                    action_str = "fail"
                
                exec_actions.append(action_str)
                actions.append(raw_action)
        return exec_actions, actions

    def get_qwen_action_raw(self, ):
        if self.ability_buffer_idx >= len(self.ability_buffer):
            self.ability_buffer = []
            ret = self.get_core_result()
            if not ret:
                return ["fail"]
            assert len(self.ability_buffer)
            self.ability_buffer_idx = 0
        
        ability_name, ability_args = self.ability_buffer[self.ability_buffer_idx]
        ability_res = self.get_ability_result(ability_name, ability_args)
        self.ability_buffer_idx += 1
        if ability_name == "exploration_guidance":
            self.last_to_find = ability_args    
            
            place = ability_res
            if place is None:
                return ["fail"]
            self.explored.append(place)
            if place.startswith("target "):
                place = "arrive at " + place[len("target "):]
            elif place.startswith("in "):
                place = "check the inside of " + place[len("in "):]
            elif place.startswith("on "):
                place = "check " + place[len("on "):]
            else:
                place = "check " + place
            self.exploration_subgoal = place
        elif ability_name == "object_grounding":
            if ability_res == False:
                if not self.core_history.strip().endswith("Grounding feedback: the target object is not found"):
                    self.core_history += "Grounding feedback: the target object is not found\n"
            else:
                ocls = ability_res[0]["label"]
                if self.env_name == "alfworld":
                    ocls = ocls.replace("sliced ", "")
                if not self.core_history.strip().endswith("Grounding feedback: the target object is not found"):
                    self.core_history += f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
                else:
                    self.core_history = self.core_history.strip()[:-len("Grounding feedback: the target object is not found")] + f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
        elif ability_name == "exploration_planner":
            steps = ability_res
            self.last_local_traj = []
            self.exploration_subgoal = None
            return steps
        elif ability_name == "manipulation_planner":
            steps = ability_res
            self.last_local_traj = []
            return steps
        elif ability_name == "scene_description":
            self.scene_description = ability_res
        elif ability_name == "experience_summarization":
            self.core_history += f"Summarization feedback: {ability_res}\n"
            self.manipulation_subgoal = None
        else:
            print(ability_name)
            raise NotImplementedError
        
        return ["pass"]
    
    def get_core_result(self,):
        res = qwen_inference(
            self.vlm_processer, self.vlm, 
            [], 
            self.prompt_ct.format(self.task_instruction, self.core_history),
            log_file=f"{self.save_path}/qwen_log.txt"
        )
        if "Query:" not in res and "query:" in res:
            res = res.replace("query:", "Query:")
        if "Query:" in res:
            think_text = res.split("Query:")[0].split("Think:")[1].strip()
            queries_text = res.split("Query:")[1].strip()
            
            if self.core_history.strip().endswith("Grounding feedback: the target object is not found") and self.core_history.strip()[:-len("Grounding feedback: the target object is not found")].strip().endswith(queries_text):
                pass
            else:
                self.core_history += "Query: " + queries_text + "\n"
            queries = queries_text.split("\n")
            queries = [q.split(". ")[1].strip() for q in queries]
            for iq, query in enumerate(queries):
                ability_name = query.split("(")[0]
                args = "(".join(query.split("(")[1:])
                assert args.endswith(")"), args
                args = args[:-1]
                self.ability_buffer.append([ability_name, args])
            return True
        else:
            if "Stop" not in res:
                return False
            think_text = res.split("Stop")[0].split("Think:")[0].strip()
            return False
        
            
    def get_ability_result(self, ability_name, args):
        if ability_name == "exploration_guidance":
            if args == self.last_to_find:
                pass
            elif self.last_to_find and args == self.last_to_find.split(" (hint:")[0]: 
                pass
            else:
                self.explored = []
            target_obj = args
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            iii = 0
            while True:
                if self.env_name == "alfworld":
                    if not(res in self.explored or res.split(" ")[0] not in ["in", "on", "target"] or " ".join(res.split(" ")[1:]).lower() not in self.observed_objects_list):
                        break
                elif self.env_name == "eb-alfred":
                    if not (res in self.explored or res.split(" ")[0] not in ["in", "on", "target"] or " ".join(res.split(" ")[1:]) not in self.observed_objects_list):
                        break
                iii += 1
                more_args = {
                    "do_sample": True,
                    "temperature": 0.8+iii*0.1,
                    "top_k": 50, 
                    "top_p": 0.9,
                }
                res = qwen_inference(
                    self.vlm_processer, self.vlm, 
                    [], 
                    self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored), more_args=more_args,
                    log_file=f"{self.save_path}/qwen_log.txt"
                ).strip().replace("{", "").replace("}", "")
                if iii > 10:
                    return None
                
            assert res not in self.explored, [self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored), res]
            return res
        elif ability_name == "object_grounding":
            target_obj = args.split(" (hint")[0].split(" (except")[0]
            if self.last_goto == target_obj: # shortcut
                return [{"label": target_obj}]
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                self.prompt_og.format(target_obj),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert (res.startswith("```json") and res.endswith("```")) or res.lower() == "no", res
            ret = eval(res[8:-3].strip()) if (res.startswith("```json") and res.endswith("```")) else False
            if ret == False:
                self.last_grounding_label = None
            else:
                self.last_grounding_label = ret[0]["label"]
            return ret 
        elif ability_name == "exploration_planner":
            assert self.exploration_subgoal
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_lpe.format(self.exploration_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            return [x.strip() for x in steps]
        elif ability_name == "manipulation_planner":
            manipulation_subgoal = args
            self.manipulation_subgoal = manipulation_subgoal
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_lpm.format(self.invent, self.last_goto, self.scene_description, manipulation_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            return [x.strip() for x in steps]
        elif ability_name == "scene_description":
            if self.invent != "nothing":
                invent_des = f" Note that the agent is holding {self.invent}, which is shown at the bottom of the image. You can ignore it in your description. "
            else:
                invent_des = ""
            # assert self.last_grounding_label is not None
            if self.last_grounding_label is None:
                return ""
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                self.prompt_sd.format(self.last_grounding_label) + invent_des,
                max_new_tokens=512,
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            return res
        elif ability_name == "experience_summarization":
            assert len(self.last_local_traj)
            assert "none" not in "\n".join(self.last_local_traj), self.last_local_traj
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                self.prompt_es.format(self.manipulation_subgoal, "\n".join(self.last_local_traj)),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            return res
        else:
            print(ability_name)
            raise NotImplementedError
        
