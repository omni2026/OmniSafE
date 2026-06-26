import os
import json

from .api_client import APIClient
from .prompts import PROMPTS


class Agent(object):
    def __init__(self, api_client, env_name="alfworld"):
        assert isinstance(api_client, APIClient)
        self.api_client = api_client
        self.env_name = env_name

        prompt_key = env_name if env_name in PROMPTS else "alfworld"
        prompts = PROMPTS[prompt_key]
        self.prompt_og = prompts["prompt_og"]
        self.prompt_sd = prompts["prompt_sd"]
        self.prompt_lpe = prompts["prompt_lpe"]
        self.prompt_lpm = prompts["prompt_lpm"]
        self.prompt_eg = prompts["prompt_eg"]
        self.prompt_es = prompts["prompt_es"]
        self.prompt_ct = prompts["prompt_ct"]

        self.trace = []

    def reset(self, save_path, obj_list):
        self.last_goto = None
        self.save_path = save_path
        self.save_i = 0

        self.observed_objects_list = [x for x in sorted(obj_list)]
        self.observed_objects_list_lower = [x.lower() for x in self.observed_objects_list]
        self.core_history = ""
        self.explored = []
        self.invent = "nothing"
        self.last_local_traj = []
        self.ability_buffer = []
        self.ability_buffer_idx = 0
        self.last_to_find = None
        self.slice_idx = 3
        self.cur_rgb_path = None
        self.last_grounding_label = None
        self.scene_description = ""
        self.exploration_subgoal = None
        self.manipulation_subgoal = None
        self.last_action = None

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "qwen_log.txt"), "w") as f:
            f.write("BEGIN!!!\n")

        self.trace = []

    def process_observation(self, rgb_or_path, env_step_id=None):
        if isinstance(rgb_or_path, str):
            self.cur_rgb_path = rgb_or_path
        else:
            try:
                import cv2
                path = os.path.join(self.save_path, f"step_{env_step_id}.png")
                cv2.imwrite(path, rgb_or_path[:, :, ::-1])
                self.cur_rgb_path = path
            except ImportError:
                raise RuntimeError(
                    "cv2 is required for numpy array observations. "
                    "Install opencv-python or pass image path directly."
                )

    def process_task(self, task_info, task_instruction):
        print("[TASK] ", task_instruction)
        self.task_instruction = task_instruction

    def process_feedback(self, message, last_action):
        if self.env_name not in ["generic"]:
            assert last_action not in ["examine", "pass", "do nothing"]

        self.last_action = last_action
        aid = len(self.last_local_traj) // 2 + 1
        self.last_local_traj.append(f"[action {aid}] {last_action}")
        self.last_local_traj.append(f"[feedback {aid}] {'success' if message else 'failure'}")

        self.scene_description = ""
        if message:
            if self.env_name in ["alfworld", "generic"]:
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

    def get_qwen_action(self, ):
        raw_actions = self.get_qwen_action_raw()

        exec_actions = []
        actions = []

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
                raw_action = raw_action.replace("Spray bottle", "SprayBottle").replace("the sponge", "the DishSponge").replace(" Sliced ", " ").replace("Floor lamp", "FloorLamp").replace("Desk lamp", "DeskLamp").replace("lettuce", "Lettuce").replace("apple", "Apple").replace("potato", "Potato").replace("tomato", "Tomato").replace("bread", "Bread").replace("the Sponge", "the DishSponge").replace("the Key ", "the KeyChain ").replace("Garbage can", "GarbageCan").replace("the towel", "the HandTowel").replace("Inkpen", "Pen").replace("Watering can", "WateringCan").replace("Armchair", "ArmChair").replace(" sliced ", " ").replace("Glass bottle", "Glassbottle").replace("Soap bottle", "SoapBottle").replace("SlicedApple", "Apple").replace("SlicedLettuce", "Lettuce").replace("SlicedBread", "Bread").replace("SlicedTomato", "Tomato").replace("SlicedPotato", "Potato").replace("GlassBottle", "Glassbottle").replace("the knife", "the Knife").replace("AppleSliced", "Apple").replace("LettuceSliced", "Lettuce").replace("BreadSliced", "Bread").replace("TomatoSliced", "Tomato").replace("PotatoSliced", "Potato")
                if raw_action.endswith("the Key"):
                    raw_action = raw_action.replace("the Key", "the KeyChain")

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
                    action_str = "fail"

                exec_actions.append(action_str)
                actions.append(raw_action)
        else:
            exec_actions = raw_actions
            actions = raw_actions

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

    def _call_api(self, image_paths, prompt, max_new_tokens=4096,
                  temperature=0.0, top_p=1.0, ability_name="unknown"):
        log_file = os.path.join(self.save_path, "qwen_log.txt")
        res = self.api_client.inference(
            image_paths=image_paths,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            log_file=log_file,
        )
        self.trace.append({
            "ability": ability_name,
            "prompt": prompt,
            "images": list(image_paths) if image_paths else [],
            "response": res,
            "temperature": temperature,
            "top_p": top_p,
        })
        return res

    def get_core_result(self, ):
        res = self._call_api(
            image_paths=[],
            prompt=self.prompt_ct.format(self.task_instruction, self.core_history),
            ability_name="cognitive_task_planner",
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
            parsed = []
            for q in queries:
                q = q.strip()
                if not q:
                    continue
                if ". " in q:
                    q = q.split(". ", 1)[1].strip()
                else:
                    q = q.lstrip("0123456789.").strip()
                if "(" in q:
                    parsed.append(q)
            if not parsed:
                return False
            for query in parsed:
                ability_name = query.split("(")[0].strip()
                args = "(".join(query.split("(")[1:])
                if not args.endswith(")"):
                    args = args.rstrip(")") + ")"
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
            res = self._call_api(
                image_paths=[],
                prompt=self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                ability_name="exploration_guidance",
            ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            iii = 0
            while True:
                valid = True
                if self.env_name == "alfworld":
                    if res in self.explored:
                        valid = False
                    elif res.split(" ")[0] not in ["in", "on", "target"]:
                        valid = False
                    elif " ".join(res.split(" ")[1:]).lower() not in self.observed_objects_list_lower:
                        valid = False
                elif self.env_name == "generic":
                    if res in self.explored:
                        valid = False
                    elif res.split(" ")[0] not in ["in", "on", "target", "near"]:
                        valid = False
                    elif " ".join(res.split(" ")[1:]).lower() not in [x.lower() for x in self.observed_objects_list]:
                        valid = False
                elif self.env_name == "eb-alfred":
                    if res in self.explored:
                        valid = False
                    elif res.split(" ")[0] not in ["in", "on", "target", "near"]:
                        valid = False
                    elif " ".join(res.split(" ")[1:]) not in self.observed_objects_list:
                        valid = False
                if valid:
                    break
                iii += 1
                res = self._call_api(
                    image_paths=[],
                    prompt=self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                    temperature=0.8 + iii * 0.1,
                    top_p=0.9,
                    ability_name="exploration_guidance_retry",
                ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
                if iii > 10:
                    return None

            assert res not in self.explored, [self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored), res]
            return res

        elif ability_name == "object_grounding":
            target_obj = args.split(" (hint")[0].split(" (except")[0]
            if self.last_goto == target_obj:
                return [{"label": target_obj}]
            if self.cur_rgb_path is None:
                return False
            res = self._call_api(
                image_paths=[self.cur_rgb_path],
                prompt=self.prompt_og.format(target_obj),
                ability_name="object_grounding",
            ).strip()
            try:
                assert (res.startswith("```json") and res.endswith("```")) or res.lower() == "no", res
                ret = eval(res[8:-3].strip()) if (res.startswith("```json") and res.endswith("```")) else False
            except (AssertionError, SyntaxError):
                ret = False
            if ret == False:
                self.last_grounding_label = None
            else:
                self.last_grounding_label = ret[0]["label"]
            return ret

        elif ability_name == "exploration_planner":
            assert self.exploration_subgoal
            res = self._call_api(
                image_paths=[],
                prompt=self.prompt_lpe.format(self.exploration_subgoal),
                ability_name="exploration_planner",
            ).strip()
            try:
                assert res.startswith("[") and res.endswith("]"), res
                steps = res[1:-1].split(",")
                assert len(steps), res
                return [x.strip() for x in steps]
            except (AssertionError, IndexError):
                return [res.strip()]

        elif ability_name == "manipulation_planner":
            manipulation_subgoal = args
            self.manipulation_subgoal = manipulation_subgoal
            res = self._call_api(
                image_paths=[],
                prompt=self.prompt_lpm.format(self.invent, self.last_goto, self.scene_description, manipulation_subgoal),
                ability_name="manipulation_planner",
            ).strip()
            try:
                assert res.startswith("[") and res.endswith("]"), res
                steps = res[1:-1].split(",")
                assert len(steps), res
                return [x.strip() for x in steps]
            except (AssertionError, IndexError):
                return [res.strip()]

        elif ability_name == "scene_description":
            if self.invent != "nothing":
                invent_des = f" Note that the agent is holding {self.invent}, which is shown at the bottom of the image. You can ignore it in your description. "
            else:
                invent_des = ""
            if self.last_grounding_label is None:
                return ""
            if self.cur_rgb_path is None:
                return ""
            res = self._call_api(
                image_paths=[self.cur_rgb_path],
                prompt=self.prompt_sd.format(self.last_grounding_label) + invent_des,
                max_new_tokens=512,
                ability_name="scene_description",
            ).strip()
            return res

        elif ability_name == "experience_summarization":
            assert len(self.last_local_traj)
            assert "none" not in "\n".join(self.last_local_traj), self.last_local_traj
            image_paths = [self.cur_rgb_path] if self.cur_rgb_path else []
            res = self._call_api(
                image_paths=image_paths,
                prompt=self.prompt_es.format(self.manipulation_subgoal, "\n".join(self.last_local_traj)),
                ability_name="experience_summarization",
            ).strip()
            return res

        else:
            print(ability_name)
            raise NotImplementedError

    def plan(self, instruction, image_path=None, obj_list=None,
             save_path=None, max_rounds=50, verbose=True):
        """
        One-call planning interface.

        Args:
            instruction: Task instruction string.
            image_path:  Path to egocentric RGB image (png/jpg).
            obj_list:    List of observable objects (e.g. ["Apple 1", "Fridge 1"]).
                         Required for exploration_guidance to validate directions.
            save_path:   Directory for logs/traces. Defaults to "api_agent_output".
            max_rounds:  Max ability-chain iterations before forcing stop.
            verbose:     Print progress.

        Returns:
            dict: {
                "actions":   list of action strings,
                "trace":     list of all API call records,
                "finished":  bool,
            }
        """
        if save_path is None:
            save_path = "api_agent_output"
        if obj_list is None:
            obj_list = []

        self.save_path = save_path
        self.reset(save_path, obj_list)
        self.process_task(None, instruction)
        if image_path is not None:
            self.process_observation(image_path)

        actions_all = []
        finished = False

        for i in range(max_rounds):
            raw = self.get_qwen_action_raw()

            if raw == ["fail"]:
                if verbose:
                    print(f"[plan] Agent gave up at round {i}.")
                break

            if raw == ["pass"]:
                if verbose and self.trace:
                    t = self.trace[-1]
                    print(f"  [{t['ability']}] {t['response'][:100].replace(chr(10),' ')}...")
                continue

            actions_all.extend(raw)
            if verbose:
                print(f"\n[ACTIONS] {raw}")

            for act in raw:
                self.process_feedback(True, act)

            if image_path is not None:
                self.process_observation(image_path)

            finished = True
            break

        self.save_trace()
        return {
            "actions": actions_all,
            "trace": self.trace,
            "finished": finished,
        }

    def plan_step(self, instruction=None, image_path=None, obj_list=None,
                  save_path=None):
        """
        Step-by-step planning interface. Call once per planning step.
        Returns actions when the agent produces them; call again after
        providing feedback + new observation.

        First call example:
            result = agent.plan_step(
                instruction="heat Apple 1 with Microwave 1",
                image_path="obs_0.png",
                obj_list=["Apple 1", "Fridge 1", ...],
            )
            # result["actions"] = ["go to Apple 1", "take Apple 1 from CounterTop 1", ...]

        Subsequent calls (after executing actions in your own env):
            agent.process_feedback(True, "go to Apple 1")
            agent.process_observation("obs_1.png")
            result = agent.plan_step()
            # result["actions"] = ["take Apple 1 from CounterTop 1", ...]

        Args:
            instruction: Task instruction. Only needed on first call.
            image_path:  Current egocentric image. Provide when available.
            obj_list:    Observable objects. Only needed on first call.
            save_path:   Log directory. Only needed on first call.

        Returns:
            dict: {"actions": [...], "finished": bool}
        """
        if not hasattr(self, 'task_instruction') or self.task_instruction is None:
            if save_path is None:
                save_path = "api_agent_output"
            if obj_list is None:
                obj_list = []
            self.save_path = save_path
            self.reset(save_path, obj_list)
            if instruction is not None:
                self.process_task(None, instruction)
        if image_path is not None:
            self.process_observation(image_path)

        for i in range(50):
            raw = self.get_qwen_action_raw()
            if raw == ["fail"]:
                return {"actions": ["fail"], "finished": False}
            if raw == ["pass"]:
                continue
            return {"actions": raw, "finished": False}

        return {"actions": ["fail"], "finished": True}

    def save_trace(self, path=None):
        if path is None:
            path = os.path.join(self.save_path, "trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboAgent API Debug")
    parser.add_argument("--task", type=str, default="heat Apple 1 with Microwave 1")
    parser.add_argument("--image", type=str, default=None, help="Path to egocentric image")
    parser.add_argument("--objects", type=str, nargs="*", default=[
        "Apple 1", "CounterTop 1", "Fridge 1", "Microwave 1",
        "Cabinet 1", "Cabinet 2", "Sink 1", "Stove 1",
    ], help="Object list")
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--env", type=str, default="alfworld", choices=["alfworld", "eb-alfred", "generic"])
    parser.add_argument("--save_path", type=str, default="api_agent_output")
    parser.add_argument("--mode", type=str, default="plan", choices=["plan", "step"],
                        help="plan=one-shot, step=interactive step-by-step")
    args = parser.parse_args()

    client_kwargs = {"model_name": args.model}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.api_key:
        client_kwargs["api_key"] = args.api_key

    api_client = APIClient(**client_kwargs)
    agent = Agent(api_client, env_name=args.env)

    if args.mode == "plan":
        result = agent.plan(
            instruction=args.task,
            image_path=args.image,
            obj_list=args.objects,
            save_path=args.save_path,
            verbose=True,
        )
        print(f"\n{'='*60}")
        print(f"RESULT: {result['actions']}")
        print(f"Finished: {result['finished']}")
        print(f"Trace saved to {args.save_path}/trace.json")

    elif args.mode == "step":
        agent.save_path = args.save_path
        step = 0
        print(f"Task: {args.task}")
        print(f"Objects: {args.objects}")
        if args.image:
            print(f"Image: {args.image}")
        print(f"Commands: 'r' = run next step, 'f <action>' = feedback, 'i <path>' = new image, 'q' = quit\n")

        r = agent.plan_step(
            instruction=args.task,
            image_path=args.image,
            obj_list=args.objects,
            save_path=args.save_path,
        )
        print(f"[Step {step}] Actions: {r['actions']}\n")
        step += 1

        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue
            if cmd == "q":
                break
            elif cmd == "r":
                r = agent.plan_step()
                print(f"[Step {step}] Actions: {r['actions']}\n")
                step += 1
            elif cmd.startswith("f "):
                parts = cmd[2:].strip().split("|")
                action = parts[0].strip()
                success = parts[1].strip().lower() == "y" if len(parts) > 1 else True
                agent.process_feedback(success, action)
                print(f"  Feedback: {action} -> {'success' if success else 'failure'}")
            elif cmd.startswith("i "):
                path = cmd[2:].strip()
                if os.path.exists(path):
                    agent.process_observation(path)
                    print(f"  Observation: {path}")
                else:
                    print(f"  File not found: {path}")
            else:
                print("  Unknown command. Use: r | f <action>|[y|n] | i <image_path> | q")

        agent.save_trace()
        print(f"\nTrace saved to {args.save_path}/trace.json")
