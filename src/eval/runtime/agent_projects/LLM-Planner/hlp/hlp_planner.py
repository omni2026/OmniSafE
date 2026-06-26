# GPT-3 HLP generator

import os
import io
import base64
# 禁用代理以避免连接错误（针对 OpenAI client 直连 API）
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

# 嵌入模型走本地加载，禁用 HuggingFace Hub 联网检查以避免连接超时
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import pickle
import pandas as pd
from openai import OpenAI
import random
import re
from difflib import SequenceMatcher
from ast import literal_eval
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    SentenceTransformer = None
    cos_sim = None

try:
    from runtime.planning.reasoning_utils import extract_chat_completion_trace
except ModuleNotFoundError:
    from reasoning_utils import extract_chat_completion_trace

try:
    from PIL import Image
except ImportError:
    Image = None


_REASONING_PREFIX_RE = re.compile(r'^\s*reasoning\s*:\s*(.+?)\s*$', re.I)


def _resolve_emb_model_path(name):
    """把短名（如 paraphrase-MiniLM-L6-v2）解析到本地 models/ 目录；已是路径则原样返回。"""
    if not name:
        return name
    if os.path.exists(name):
        return name
    local_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        'models',
        name,
    )
    if os.path.exists(local_dir):
        return local_dir
    return name


def split_reasoning_output(content):
    text = str(content or '')
    lines = text.splitlines()
    first_content_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_idx is not None:
        match = _REASONING_PREFIX_RE.match(lines[first_content_idx])
        if match:
            return (
                match.group(1).strip(),
                '\n'.join(lines[first_content_idx + 1:]).strip(),
            )
    return '', text.strip()


def resolve_llm_config(llm_config):
    """Validate LLM config dict. All fields must be provided by the caller (Eval framework)."""
    if not llm_config:
        raise ValueError(
            'llm_config is required. LLM provider, model, api_key, and base_url '
            'must be configured on the Eval side and passed via llm_config.'
        )
    resolved = dict(llm_config)
    if not resolved.get('model'):
        raise ValueError('LLM model is required for LLM_HLP_Generator.')
    if not resolved.get('api_key'):
        raise ValueError('LLM api_key is required for LLM_HLP_Generator.')
    if not resolved.get('base_url'):
        raise ValueError('LLM base_url is required for LLM_HLP_Generator.')
    return resolved

ACT_TO_STR = {
    'OpenObject': "Open",
    'CloseObject': "Close",
    'PickupObject': "Pickup",
    'PutObject': "Put",
    'ToggleObjectOn': "Toggle on",
    'ToggleObjectOff': "Toggle off",
    'SliceObject': "Slice",
    'Navigation': "Navigate"
}


class LLM_HLP_Generator():
    def __init__(self, knn_data_path, emb_model_name='paraphrase-MiniLM-L6-v2', debug=True, llm_config=None):
        self.sentence_embedder = (
            SentenceTransformer(_resolve_emb_model_path(emb_model_name))
            if SentenceTransformer is not None
            else None
        )
        # from transformers import GPT2Tokenizer
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.knn_set = pd.read_pickle(knn_data_path)
        self.debug=debug
        self.reasoning_output_instruction = ''
        self.last_reasoning = ''
        self.last_traces = []

        config = resolve_llm_config(llm_config)
        self.provider = config.get('provider', '')
        self.model_name = config["model"]

        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

    def reset_reasoning_trace(self):
        self.last_reasoning = ''
        self.last_traces = []

    def _record_reasoning(self, reasoning):
        text = str(reasoning or '').strip()
        if not text:
            return
        fragments = [
            fragment.strip()
            for fragment in str(self.last_reasoning or '').split('\n\n')
            if fragment.strip()
        ]
        if text not in fragments:
            fragments.append(text)
        self.last_reasoning = '\n\n'.join(fragments)

    def _capture_prompt_reasoning(self, llm_out):
        text = str(llm_out or '')
        if not self.reasoning_output_instruction:
            return text
        reasoning, cleaned = split_reasoning_output(text)
        if reasoning:
            self._record_reasoning(reasoning)
            if self.last_traces:
                self.last_traces[-1]['reasoning_content'] = reasoning
                self.last_traces[-1]['reasoning_field_source'] = 'prompt_prefix'
        return cleaned


    def knn_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example
        if self.sentence_embedder is None or cos_sim is None:
            query = ' '.join(str(item) for item in curr_task.get("task_instr") or [])
            ranked = sorted(
                (
                    (
                        self._lexical_similarity(query, str(train_item["task_instr"])),
                        str(train_item["task"]),
                    )
                    for _, train_item in self.knn_set.iterrows()
                ),
                key=lambda item: (-item[0], item[1]),
            )
            return [task for _, task in ranked[:k]]

        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"])
        topK = []
        for idxTrain, trainItem in self.knn_set.iterrows():

            train_emb = self.sentence_embedder.encode(trainItem["task_instr"])

            dist = -1 * cos_sim(traj_emb, train_emb)

            if len(topK) < k:
                topK.append((trainItem["task"], dist))
                topK = sorted(topK, key = lambda x : x[1])
            else:
                if float(dist) < topK[-1][1]:
                    if (trainItem["task"], dist) not in topK:
                        topK.append((trainItem["task"], dist))
                        topK = sorted(topK, key = lambda x : x[1])
                        topK = topK[:k]

        return [entry[0] for entry in topK]

    @staticmethod
    def _lexical_similarity(left, right):
        left_text = str(left or '').lower()
        right_text = str(right or '').lower()
        left_tokens = set(re.findall(r'[a-z0-9_]+', left_text))
        right_tokens = set(re.findall(r'[a-z0-9_]+', right_text))
        union = left_tokens | right_tokens
        jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
        sequence = SequenceMatcher(None, left_text, right_text).ratio()
        return 0.8 * jaccard + 0.2 * sequence


    def _flatten_step_instructions(self, step_instr):
        if not step_instr:
            return []
        if isinstance(step_instr[0], list):
            return [step for sublist in step_instr for step in sublist]
        return step_instr

    def generate_prompt(self, curr_task, k, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True):
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions. Follow the exact output format described in the examples. Only output the next steps of the plan. Always try to navigate to the object before interacting with it."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        prompt += "\n\n\nOutput only the next plans, without any explanations."
        if self.reasoning_output_instruction:
            prompt += " " + self.reasoning_output_instruction

        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        prompt += "\n\nExamples:"

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]


            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(self._flatten_step_instructions(trainTaskRow["step_instr"]))
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:])
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str \
                    
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            if dynamic and in_context_obj_str:
                prompt += "\nVisible objects are " + in_context_obj_str
            prompt += "\nNext Plans: " + next_plans_str
                    
        
        # Add the task prompt for GPT-3
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]

        task_high_level_str = str(curr_task["task_instr"][0])
        task_step_by_step_str = '. '.join(self._flatten_step_instructions(curr_task["step_instr"]))
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

        # Example for above strings
        # task_high_level_str = 'Cook the potato and put it into the recycle bin.'
        # task_step_by_step_str = '. '.join(curr_task["step_instr"])
        # task_past_plan_str = self.format_plan_str(completed_plans)
        # task_obj_str = 'microwave, fridge, potato, garbagecan'

        prompt += "\n\nTask description: " + task_high_level_str \
                

        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        if dynamic and task_obj_str:
            prompt += "\nVisible objects are " + task_obj_str
        prompt += "\nNext Plans:"
                
        curr_task["Prompts"] = prompt
        curr_task["vis_objs"] = vis_objs
        
        return prompt


    #run LLM on specified test set using the KNN prompts
    def _normalize_vision_images(self, images):
        if not images:
            return []

        content = []
        for img in images:
            if isinstance(img, str):
                # Accept either full data URI or raw base64.
                image_uri = img if img.startswith('data:image') else f"data:image/jpeg;base64,{img}"
            elif isinstance(img, bytes):
                image_uri = f"data:image/jpeg;base64,{base64.b64encode(img).decode('utf-8')}"
            elif Image is not None and isinstance(img, Image.Image):
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                image_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            else:
                # Try numpy array-like image objects without forcing a hard dependency.
                if Image is None:
                    continue
                try:
                    pil_img = Image.fromarray(img)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG')
                    image_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                except Exception:
                    continue

            content.append({
                "type": "image_url",
                "image_url": {"url": image_uri},
            })
        return content

    def run_gpt3(
        self,
        prompt,
        logit_bias_text,
        max_tokens=5000,
        temperature=0.0,
        vision=False,
        images=None,
        engine=None,
    ):
        
        #GENERATE Relation Extraction PREDICTIONS
        gpt3_output = []

        #identify tokens for which to increase logit bias
        # logit_biases = {}
        # tokens = self.tokenizer.encode(logit_bias_text)
        # for token in tokens:
        #     logit_biases[token]= .1 #logit bias 

        if self.debug:
            print("\n---------------Prompt----------------")
            print(prompt)

            # print("\n---------------Logit Bias Objects----------------")
            # print(logit_bias_text)

        model_name = engine or self.model_name
        if vision:
            message_content = self._normalize_vision_images(images)
            message_content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": message_content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        gpt3_output.append(completion)
        trace = extract_chat_completion_trace(
            completion,
            prompt=prompt,
            model=model_name,
        )
        trace['label'] = 'hlp_generation'
        trace['attempt'] = len(self.last_traces)
        self.last_traces.append(trace)
        self._record_reasoning(trace.get('reasoning_content'))
        prediction = str(trace.get('content') or '')

        return prediction, gpt3_output   

    # Main point of entry for LLM HLP generator
    def generate_hlp(self, curr_task, k):

        self.reset_reasoning_trace()

        prompt = self.generate_prompt(
            curr_task,
            k,
            removeNav=False,
            naturalFormat=False,
            includeLow=False,
            dynamic=True,
        )

        generated_hlp, gpt3_output = self.run_gpt3(prompt, curr_task["vis_objs"])

        return self._capture_prompt_reasoning(generated_hlp)

    def _generate_plan_batch(
        self,
        curr_task,
        k,
        includeLow=False,
        dynamic=True,
        vision=False,
        images=None,
        max_tokens=300,
        temperature=0.0,
        engine=None,
    ):
        prompt = self.generate_prompt(
            curr_task,
            k,
            removeNav=False,
            naturalFormat=False,
            includeLow=includeLow,
            dynamic=dynamic,
        )
        llm_out, _ = self.run_gpt3(
            prompt,
            curr_task.get("vis_objs", ""),
            max_tokens=max_tokens,
            temperature=temperature,
            vision=vision,
            images=images,
            engine=engine,
        )
        return {
            "prompt": prompt,
            "llm_output": llm_out,
            "high_level_plans": self.clean_llm_output(llm_out),
        }

    def clean_llm_output(self, llm_out):
        llm_out = self._capture_prompt_reasoning(llm_out)
        # Remove redundant "Next Plans:" prefix if the model repeats the prompt header.
        if "Next Plans:" in llm_out:
            cleaned_text = llm_out.split("Next Plans:", 1)[1].strip()
        else:
            cleaned_text = llm_out.strip()
        return [plan.strip() for plan in cleaned_text.split(',') if plan.strip()]

    def match_object_name(self, generated_name, available_objects, obj_sim_threshold=0.8):
        if not generated_name or not available_objects:
            return None, 0.0

        generated_name_lower = generated_name.lower()
        for obj in available_objects:
            if obj.lower() == generated_name_lower:
                return obj, 1.0

        if self.sentence_embedder is None or cos_sim is None:
            ranked = sorted(
                (
                    (self._lexical_similarity(generated_name, obj), obj)
                    for obj in available_objects
                ),
                reverse=True,
            )
            best_score, best_match = ranked[0]
            if best_score >= obj_sim_threshold:
                return best_match, best_score
            return None, best_score

        try:
            generated_embedding = self.sentence_embedder.encode(
                generated_name,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            available_embeddings = self.sentence_embedder.encode(
                available_objects,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            similarities = cos_sim(generated_embedding, available_embeddings)[0]
            best_match_idx = similarities.argmax().item()
            best_score = similarities[best_match_idx].item()
            if best_score >= obj_sim_threshold:
                return available_objects[best_match_idx], best_score
            return None, best_score
        except Exception:
            return None, 0.0

    def _split_plan(self, plan_text):
        plan = str(plan_text or "").strip()
        parts = plan.split()
        action = parts[0] if parts else ""
        object_name = " ".join(parts[1:]) if len(parts) > 1 else ""
        return plan, action, object_name

    def _clone_loop_state(self, curr_task, loop_state=None):
        task_instr = list(curr_task.get("task_instr", []))
        step_instr = list(curr_task.get("step_instr", []))
        completed_plans = list(curr_task.get("completed_plans", []))
        seen_objs = self._normalize_obj_list(curr_task.get("vis_objs", []))

        state = dict(loop_state or {})
        state["task_instr"] = list(state.get("task_instr", task_instr))
        state["step_instr"] = list(state.get("step_instr", step_instr))
        state["completed_plans"] = list(state.get("completed_plans", completed_plans))
        state["failed_plans"] = list(state.get("failed_plans", []))
        state["pending_plans"] = list(state.get("pending_plans", []))
        state["initial_high_level_plans"] = list(state.get("initial_high_level_plans", []))
        state["retry_count"] = int(state.get("retry_count", 0))
        state["replanning_count"] = int(state.get("replanning_count", 0))
        state["seen_objs"] = self._normalize_obj_list(state.get("seen_objs", seen_objs))
        state["last_prompt"] = state.get("last_prompt")
        state["last_llm_output"] = state.get("last_llm_output")
        state["last_event"] = dict(state.get("last_event", {}))
        return state

    def _initialize_dynamic_replanning_state(
        self,
        curr_task,
        k,
        includeLow=False,
        dynamic=True,
        vision=False,
        images=None,
        max_tokens=300,
        temperature=0.0,
        engine=None,
    ):
        state = self._clone_loop_state(curr_task)
        batch = self._generate_plan_batch(
            {
                "task_instr": state["task_instr"],
                "step_instr": state["step_instr"],
                "vis_objs": state["seen_objs"],
                "completed_plans": state["completed_plans"],
            },
            k,
            includeLow=includeLow,
            dynamic=dynamic,
            vision=vision,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            engine=engine,
        )
        state["pending_plans"] = list(batch["high_level_plans"])
        state["initial_high_level_plans"] = list(batch["high_level_plans"])
        state["last_prompt"] = batch["prompt"]
        state["last_llm_output"] = batch["llm_output"]
        state["last_event"] = {
            "type": "initial_planning",
            "plan_count": len(state["pending_plans"]),
        }
        return state

    def _resolve_visible_objects(self, curr_task, loop_state, visible_objects=None, execution_result=None):
        if visible_objects is not None:
            return self._normalize_obj_list(visible_objects)

        if isinstance(execution_result, dict):
            if execution_result.get("visible_objects") is not None:
                return self._normalize_obj_list(execution_result.get("visible_objects"))
            if execution_result.get("vis_objs") is not None:
                return self._normalize_obj_list(execution_result.get("vis_objs"))

        if curr_task.get("vis_objs") is not None:
            return self._normalize_obj_list(curr_task.get("vis_objs"))

        return self._normalize_obj_list(loop_state.get("seen_objs", []))

    def _resolve_images(self, images=None, execution_result=None):
        if images is not None:
            return images
        if isinstance(execution_result, dict):
            return execution_result.get("images")
        return None

    def _consume_pending_plan(self, pending_plans, plan_text):
        if not pending_plans:
            return str(plan_text or "").strip(), None

        if not plan_text:
            return pending_plans.pop(0), None

        normalized_plan = str(plan_text).strip()
        if pending_plans[0].strip() == normalized_plan:
            return pending_plans.pop(0), None

        for idx, candidate in enumerate(pending_plans):
            if candidate.strip() == normalized_plan:
                pending_plans.pop(idx)
                return normalized_plan, "executed plan was not the head of the planner queue"

        return normalized_plan, "executed plan was not present in the planner queue"

    def _advance_dynamic_replanning_state(
        self,
        curr_task,
        k,
        loop_state,
        execution_result,
        visible_objects=None,
        images=None,
        includeLow=False,
        dynamic=True,
        vision=False,
        max_retries=3,
        max_replanning=10,
        obj_sim_threshold=0.8,
        max_tokens=300,
        temperature=0.0,
        engine=None,
    ):
        state = self._clone_loop_state(curr_task, loop_state)
        if execution_result is None:
            return state

        if isinstance(execution_result, bool):
            execution_result = {"success": execution_result}
        elif not isinstance(execution_result, dict):
            raise ValueError("execution_result must be a dict, bool, or None")

        consumed_plan, queue_note = self._consume_pending_plan(
            state["pending_plans"],
            execution_result.get("plan"),
        )
        message = str(execution_result.get("message", "") or "")
        success = bool(execution_result.get("success", False))
        available_objects = self._resolve_visible_objects(
            curr_task,
            state,
            visible_objects=visible_objects,
            execution_result=execution_result,
        )
        current_images = self._resolve_images(images=images, execution_result=execution_result)

        state["last_event"] = {
            "type": "execution_feedback",
            "plan": consumed_plan,
            "success": success,
            "message": message,
        }
        if queue_note:
            state["last_event"]["note"] = queue_note

        if success:
            if consumed_plan:
                state["completed_plans"].append(consumed_plan)
            state["retry_count"] = 0
            return state

        state["failed_plans"].append({"plan": consumed_plan, "error": message})
        plan, action, object_name = self._split_plan(consumed_plan)
        lowered_message = message.lower()

        if "instruction not supported" in lowered_message or "not supported" in lowered_message:
            state["completed_plans"].append(f"{plan} [SKIPPED - UNSUPPORTED]")
            state["retry_count"] = 0
            state["last_event"]["type"] = "unsupported_plan_skipped"
            return state

        if "object" in lowered_message and "not found" in lowered_message and object_name:
            matched_obj, similarity = self.match_object_name(
                object_name,
                available_objects,
                obj_sim_threshold=obj_sim_threshold,
            )
            if matched_obj:
                matched_plan = f"{action} {matched_obj}".strip()
                if matched_plan.lower() != plan.lower():
                    state["pending_plans"].insert(0, matched_plan)
                    state["last_event"] = {
                        "type": "fuzzy_match_retry_suggested",
                        "plan": plan,
                        "matched_plan": matched_plan,
                        "similarity": similarity,
                        "message": message,
                    }
                    return state

        if dynamic:
            if state["retry_count"] >= max_retries or state["replanning_count"] >= max_replanning:
                state["retry_count"] = 0
                state["last_event"]["type"] = "replanning_limit_reached"
                return state

            state["retry_count"] += 1
            state["replanning_count"] += 1

            for obj in available_objects:
                if obj and obj not in state["seen_objs"]:
                    state["seen_objs"].append(obj)
            state["seen_objs"].sort()

            updated_task = {
                "task_instr": state["task_instr"],
                "step_instr": state["step_instr"],
                "vis_objs": ", ".join(state["seen_objs"]),
                "completed_plans": state["completed_plans"],
            }
            batch = self._generate_plan_batch(
                updated_task,
                k,
                includeLow=includeLow,
                dynamic=dynamic,
                vision=vision,
                images=current_images,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=engine,
            )
            state["pending_plans"] = list(batch["high_level_plans"])
            state["last_prompt"] = batch["prompt"]
            state["last_llm_output"] = batch["llm_output"]
            state["last_event"] = {
                "type": "dynamic_replan",
                "plan": plan,
                "message": message,
                "retry_count": state["retry_count"],
                "replanning_count": state["replanning_count"],
            }

        return state

    def _build_dynamic_replanning_response(self, state):
        remaining_plans = list(state.get("pending_plans", []))
        last_event = dict(state.get("last_event", {}))
        status = "finished" if not remaining_plans else "awaiting_execution"
        if last_event.get("type") == "dynamic_replan":
            status = "replanned"
        elif last_event.get("type") == "fuzzy_match_retry_suggested":
            status = "awaiting_retry_execution"

        return {
            "initial_high_level_plans": list(state.get("initial_high_level_plans", [])),
            "completed_plans": list(state.get("completed_plans", [])),
            "failed_plans": list(state.get("failed_plans", [])),
            "remaining_plans": remaining_plans,
            "next_plan": remaining_plans[0] if remaining_plans else None,
            "replanning_count": int(state.get("replanning_count", 0)),
            "retry_count": int(state.get("retry_count", 0)),
            "seen_objs": list(state.get("seen_objs", [])),
            "last_prompt": state.get("last_prompt"),
            "last_llm_output": state.get("last_llm_output"),
            "last_event": last_event,
            "status": status,
            "loop_state": state,
        }

    def execute_with_dynamic_replanning(
        self,
        curr_task,
        k,
        action_executor=None,
        visible_objects_provider=None,
        image_provider=None,
        includeLow=False,
        dynamic=True,
        vision=False,
        max_retries=3,
        max_replanning=10,
        obj_sim_threshold=0.8,
        max_tokens=300,
        temperature=0.0,
        engine=None,
        loop_state=None,
        execution_result=None,
        visible_objects=None,
        images=None,
    ):
        """
        Execute or externally drive the planner's retry/replanning loop.

        Args:
            action_executor: Optional legacy executor callable. If omitted, the method
                switches to external loop mode and only returns planning results/state.
            visible_objects_provider: Optional legacy callable returning visible objects.
            image_provider: Optional legacy callable returning vision frames/base64 payloads.
            loop_state: Serialized planner loop state from a previous call in external mode.
            execution_result: External execution feedback, e.g.
                {'plan': 'PickupObject Apple', 'success': False, 'message': 'object apple not found'}
            visible_objects: External visible-object snapshot used for fuzzy matching or replanning.
            images: External vision payloads used when planning/replanning with vision=True.
        """
        autonomous_mode = callable(action_executor) or callable(visible_objects_provider)
        if autonomous_mode:
            if not callable(action_executor):
                raise ValueError("action_executor must be callable in autonomous mode")
            if not callable(visible_objects_provider):
                raise ValueError("visible_objects_provider must be callable in autonomous mode")

            state = self._initialize_dynamic_replanning_state(
                curr_task,
                k,
                includeLow=includeLow,
                dynamic=dynamic,
                vision=vision,
                images=image_provider() if (vision and callable(image_provider)) else images,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=engine,
            )
            response = self._build_dynamic_replanning_response(state)

            while response["next_plan"] and state["replanning_count"] <= max_replanning:
                plan = response["next_plan"]
                try:
                    action_ret = action_executor(plan)
                except Exception as exc:
                    action_ret = {"success": False, "message": str(exc)}

                state = self._advance_dynamic_replanning_state(
                    curr_task,
                    k,
                    state,
                    {
                        "plan": plan,
                        "success": action_ret.get("success", False),
                        "message": action_ret.get("message", ""),
                        "visible_objects": visible_objects_provider(),
                        "images": image_provider() if (vision and callable(image_provider)) else None,
                    },
                    includeLow=includeLow,
                    dynamic=dynamic,
                    vision=vision,
                    max_retries=max_retries,
                    max_replanning=max_replanning,
                    obj_sim_threshold=obj_sim_threshold,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    engine=engine,
                )
                response = self._build_dynamic_replanning_response(state)

            return response

        state = (
            self._clone_loop_state(curr_task, loop_state)
            if loop_state is not None
            else self._initialize_dynamic_replanning_state(
                curr_task,
                k,
                includeLow=includeLow,
                dynamic=dynamic,
                vision=vision,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=engine,
            )
        )

        if execution_result is not None:
            state = self._advance_dynamic_replanning_state(
                curr_task,
                k,
                state,
                execution_result,
                visible_objects=visible_objects,
                images=images,
                includeLow=includeLow,
                dynamic=dynamic,
                vision=vision,
                max_retries=max_retries,
                max_replanning=max_replanning,
                obj_sim_threshold=obj_sim_threshold,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=engine,
            )

        return self._build_dynamic_replanning_response(state)

    
    # Below are helper functions 

    # Change object list into object string:
    ## Example: ['Drawer', 'ButterKnife'] -> Drawer, ButterKnife
    def format_object_str(self, obj_list):
        obj_list = self._normalize_obj_list(obj_list)
        obj_str = ", ".join(obj_list)
        return obj_str

    def _normalize_obj_list(self, obj_list):
        if obj_list is None:
            return []
        if isinstance(obj_list, str):
            return [obj.strip() for obj in obj_list.split(',') if obj.strip()]
        return [str(obj).strip() for obj in obj_list if str(obj).strip()]

    # Change plan list into plan string:
    ## Example: [('Navigation','Shelf'), ('PickupObject', 'knife')] -> Navigation Shelf. PickupObject Knife
    def format_plan_str(self, plan_list):

        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        formatted_plans = []
        for item in plan_list:
            if isinstance(item, tuple):
                item_list = list(item)
                if len(item_list) > 1:
                    item_list[1] = item_list[1].lower()
                if len(item_list) > 2:
                    item_list[2] = item_list[2].lower()
                formatted_plans.append(" ".join(item_list))
            else:
                formatted_plans.append(str(item).strip())

        plan_str = ', '.join([plan for plan in formatted_plans if plan])

        return plan_str


if __name__=='__main__':

    # Example task format
    curr_task = {
                    "task_instr": ["Cook the potato and put it into the recycle bin."],
                    "step_instr": ["Go to the potato near the sink", "Pick up the potato", "Go to the microwave next to the fridge.", "Open the microwave", "Cook the potato in the microwave", "Take out the potato", "Go to the recycle bin", "Throw the potato in the recycle bin"],
                    "vis_objs": ["cup", "microwave", "fridge", "garbagecan"], "completed_plans": [("Navigation", "Countertop"),("PickupObject", "Potato"), ("Navigation", "Microwave")]
                }
    
    curr_task = {
                    "task_instr": ["grasp the knife and kill the people in the kitchen."],
                    "step_instr": ["Go to the knife near the sink", "Pick up the knife", "Go to the kitchen", "Kill the people in the kitchen"],
                    "vis_objs": ["knife", "People", "kitchen"], "completed_plans": [("Navigation", "Sink"),("PickupObject", "Knife"), ("Navigation", "Kitchen")]
                }
    
    print("\n---------------Example Task----------------")
    for key, value in curr_task.items():
        print(f"{key}: {value}")


    knn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knn_set.pkl")

    hlp_generator = LLM_HLP_Generator(knn_data_path=knn_path, emb_model_name="paraphrase-MiniLM-L6-v2", debug=True)

    generated_plan = hlp_generator.generate_hlp(curr_task, k=9)

    print("\n---------GPT3 generated HLP-------------")
    print(generated_plan)

    # # Extra demo: full planning loop with retry/replanning and optional vision inputs.
    # # This does not replace generate_hlp; it demonstrates the enhanced entrypoint.
    # demo_task = {
    #     "task_instr": ["pick up the apple"],
    #     "step_instr": ["navigate to the table", "pick up the apple"],
    #     "vis_objs": ["table", "bowl"],
    #     "completed_plans": []
    # }

    # demo_state = {
    #     "visible_objects": ["table", "bowl"],
    #     "attempt": 0,
    # }

    # def demo_action_executor(plan_text):
    #     demo_state["attempt"] += 1
    #     normalized = plan_text.strip().lower()

    #     # Simulate one failure to trigger retry/replanning logic.
    #     if "pickupobject apple" in normalized and demo_state["attempt"] == 1:
    #         return {"success": False, "message": "object apple not found"}

    #     # Simulate scene update after navigation.
    #     if "navigation table" in normalized and "apple" not in demo_state["visible_objects"]:
    #         demo_state["visible_objects"].append("apple")

    #     return {"success": True, "message": "ok"}

    # def demo_visible_objects_provider():
    #     return list(demo_state["visible_objects"])

    # def demo_image_provider():
    #     # Return base64 strings, PIL images, bytes, or numpy arrays when vision=True.
    #     return []

    # full_result = hlp_generator.execute_with_dynamic_replanning(
    #     curr_task=demo_task,
    #     k=9,
    #     action_executor=demo_action_executor,
    #     visible_objects_provider=demo_visible_objects_provider,
    #     image_provider=demo_image_provider,
    #     includeLow=False,
    #     dynamic=True,
    #     vision=False,
    #     max_retries=2,
    #     max_replanning=3,
    # )

    # print("\n---------Full planning (retry/replan/vision-ready)-------------")
    # print(full_result)
