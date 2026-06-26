"""
Code-as-Policy: Configurable Language Model Program Implementation
Translates natural language instructions to executable Python code

Core Features:
- Support for multiple LLM providers (OpenAI, DeepSeek, Qwen, ZhipuAI, etc.)
- Flexible prompt template system
- Code generation and optional execution
- Dynamic function generation
- Customizable actions and utility functions
"""

import os
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, Callable, Optional, List
from time import sleep
import ast
import re

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

# astunparse is optional dependency for function extraction
try:
    import astunparse
    HAS_ASTUNPARSE = True
except ImportError:
    HAS_ASTUNPARSE = False


def _load_project_dotenv() -> None:
    if load_dotenv is None:
        return

    candidates = [
        Path(__file__).resolve().parents[3] / '.env',
        Path.cwd() / '.env',
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        load_dotenv(dotenv_path=candidate, override=False)
        seen.add(candidate)


def _resolve_api_config(provider: str) -> Dict[str, str]:
    config = dict(API_CONFIGS[provider])
    env_name = str(config.get('api_key_env', '') or '').strip()
    if env_name:
        config['api_key'] = os.getenv(env_name, str(config.get('api_key', '') or ''))
    else:
        config['api_key'] = str(config.get('api_key', '') or '')
    return config


_load_project_dotenv()


# ============================================================================
# API Configuration
# ============================================================================

API_CONFIGS = {
    "zhipuai": {
        "api_key_env": "ZHIPUAI_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_name": "glm-4.6v",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
    }
}

PROVIDER = "deepseek"  # Choose your model provider


# ============================================================================
# Prompt Templates (Based on original Code-as-Policy)
# ============================================================================

# Main task planning prompt template

# 注意，这里是针对桌面操作任务的ICL示例。
# env_utils是作者在仿真环境中实现的API；plan_utils是一些子的LMP，调用即代表进行一次LLM query
# 
DEFAULT_TASK_PROMPT = """
# Python 2D robot control script
import numpy as np
from env_utils import put_first_on_second, get_obj_pos, get_obj_names, say, get_corner_name, get_side_name, is_obj_visible, stack_objects_in_order
from plan_utils import parse_obj_name, parse_position, parse_question, transform_shape_pts
{imports}

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# the yellow block on the yellow bowl.
say('Ok - putting the yellow block on the yellow bowl')
put_first_on_second('yellow block', 'yellow bowl')

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# which block did you move.
say('I moved the yellow block')

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# move the green block to the top right corner.
say('Got it - putting the green block on the top right corner')
corner_pos = parse_position('top right corner')
put_first_on_second('green block', corner_pos)

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# stack the blue bowl on the yellow bowl on the green block.
order_bottom_to_top = ['green block', 'yellow block', 'blue bowl']
say(f'Sure - stacking from top to bottom: {", ".join(order_bottom_to_top)}')
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# move the cyan block into its corresponding bowl.
matches = {'cyan block': 'cyan bowl'}
say('Got it - placing the cyan block on the cyan bowl')
for first, second in matches.items():
  put_first_on_second(first, get_obj_pos(second))

objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# make a line of blocks on the right side.
say('No problem! Making a line of blocks on the right side')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
line_pts = parse_position(f'a 30cm vertical line on the right with {len(block_names)} points')
for block_name, pt in zip(block_names, line_pts):
  put_first_on_second(block_name, pt)

objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# put the small banana colored thing in between the blue bowl and green block.
say('Sure thing - putting the yellow block between the blue bowl and the green block')
target_pos = parse_position('a point in the middle betweeen the blue bowl and the green block')
put_first_on_second('yellow block', target_pos)

objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# can you cut the bowls in half.
say('no, I can only move objects around')

objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# stack the blocks on the right side with the gray one on the bottom.
say('Ok. stacking the blocks on the right side with the gray block on the bottom')
right_side = parse_position('the right side')
put_first_on_second('gray block', right_side)
order_bottom_to_top = ['gray block', 'green block', 'yellow block']
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# hide the blue bowl.
bowl_name = np.random.choice(['yellow bowl', 'green bowl'])
say(f'Sounds good! Hiding the blue bowl under the {bowl_name}')
put_first_on_second(bowl_name, 'blue bowl')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# stack everything with the green block on top.
say('Ok! Stacking everything with the green block on the top')
order_bottom_to_top = ['blue bowl', 'pink bowl', 'green bowl', 'pink block', 'blue block', 'green block']
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the grass-colored bowl to the left.
say('Sure - moving the green bowl left by 10 centimeters')
left_pos = parse_position('a point 10cm left of the green bowl')
put_first_on_second('green bowl', left_pos)

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# why did you move the red bowl.
say(f'I did not move the red bowl')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# undo that.
say('Sure - moving the green bowl right by 10 centimeters')
left_pos = parse_position('a point 10cm right of the green bowl')
put_first_on_second('green bowl', left_pos)

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the top most block to the corner closest to the bottom most block.
top_block_name = parse_obj_name('top most block', f'objects = {get_obj_names()}')
bottom_block_name = parse_obj_name('bottom most block', f'objects = {get_obj_names()}')
closest_corner_pos = parse_position(f'the corner closest to the {bottom_block_name}', f'objects = {get_obj_names()}')
say(f'Putting the {top_block_name} on the {get_corner_name(closest_corner_pos)}')
put_first_on_second(top_block_name, closest_corner_pos)

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the brown bowl to the side closest to the green block.
closest_side_position = parse_position('the side closest to the green block')
say(f'Got it - putting the brown bowl on the {get_side_name(closest_side_position)}')
put_first_on_second('brown bowl', closest_side_position)

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the green block to the right of the bowl that has the blue block.
bowl_name = parse_obj_name('the bowl that has the blue block', f'objects = {get_obj_names()}')
if bowl_name:
  target_pos = parse_position(f'a point 10cm to the right of the {bowl_name}')
  say(f'No problem - placing the green block to the right of the {bowl_name}')
  put_first_on_second('green block', target_pos)
else:
  say('There are no bowls that has the blue block')

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the blue block in the empty bowl.
empty_bowl_name = parse_obj_name('the empty bowl', f'objects = {get_obj_names()}')
if empty_bowl_name:
  say(f'Ok! Putting the blue block on the {empty_bowl_name}')
  put_first_on_second('blue block', empty_bowl_name)
else:
  say('There are no empty bowls')

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the other blocks to the bottom corners.
block_names = parse_obj_name('blocks other than the blue block', f'objects = {get_obj_names()}')
corners = parse_position('the bottom corners')
for block_name, pos in zip(block_names, corners):
  put_first_on_second(block_name, pos)

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the red bowl a lot to the left of the blocks.
say('Sure! Moving the red bowl to a point left of the blocks')
left_pos = parse_position('a point 20cm left of the blocks')
put_first_on_second('red bowl', left_pos)

objects = ['pink block', 'gray block', 'orange block']
# move the pinkish colored block on the bottom side.
say('Ok - putting the pink block on the bottom side')
bottom_side_pos = parse_position('the bottom side')
put_first_on_second('pink block', bottom_side_pos)

objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# is the blue block to the right of the yellow bowl?
if parse_question('is the blue block to the right of the yellow bowl?', f'objects = {get_obj_names()}'):
  say('yes, there is a blue block to the right of the yellow bow')
else:
  say('no, there is\'t a blue block to the right of the yellow bow')

objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# how many yellow objects are there?
n_yellow_objs = parse_question('how many yellow objects are there', f'objects = {get_obj_names()}')
say(f'there are {n_yellow_objs} yellow object')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the left most block to the green bowl.
left_block_name = parse_obj_name('left most block', f'objects = {get_obj_names()}')
say(f'Moving the {left_block_name} on the green bowl')
put_first_on_second(left_block_name, 'green bowl')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the other blocks to different corners.
block_names = parse_obj_name(f'blocks other than the {left_block_name}', f'objects = {get_obj_names()}')
corners = parse_position('the corners')
say(f'Ok - moving the other {len(block_names)} blocks to different corners')
for block_name, pos in zip(block_names, corners):
  put_first_on_second(block_name, pos)

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# is the pink block on the green bowl.
if parse_question('is the pink block on the green bowl', f'objects = {get_obj_names()}'):
  say('Yes - the pink block is on the green bowl.')
else:
  say('No - the pink block is not on the green bowl.')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl.
left_block_names =  parse_question('what are the blocks left of the green bowl', f'objects = {get_obj_names()}')
if len(left_block_names) > 0:
  say(f'These blocks are left of the green bowl: {", ".join(left_block_names)}')
else:
  say('There are no blocks left of the green bowl')

objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# if you see a purple bowl put it on the blue bowl
if is_obj_visible('purple bowl'):
  say('Putting the purple bowl on the pink bowl')
  put_first_on_second('purple bowl', 'pink bowl')
else:
  say('I don\'t see a purple bowl')

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# imagine that the bowls are different biomes on earth and imagine that the blocks are parts of a building.
say('ok')

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# now build a tower in the grasslands.
order_bottom_to_top = ['green bowl', 'blue block', 'green block', 'yellow block']
say('stacking the blocks on the green bowl')
stack_objects_in_order(object_names=order_bottom_to_top)

objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# show me what happens when the desert gets flooded by the ocean.
say('putting the yellow bowl on the blue bowl')
put_first_on_second('yellow bowl', 'blue bowl')

objects = ['pink block', 'gray block', 'orange block']
# move all blocks 5cm toward the top.
say('Ok - moving all blocks 5cm toward the top')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name in block_names:
  target_pos = parse_position(f'a point 5cm above the {block_name}')
  put_first_on_second(block_name, target_pos)

objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make a triangle of blocks in the middle.
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
triangle_pts = parse_position(f'a triangle with size 10cm around the middle with {len(block_names)} points')
say('Making a triangle of blocks around the middle of the workspace')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(block_name, pt)

objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make the triangle smaller.
triangle_pts = transform_shape_pts('scale it by 0.5x', shape_pts=triangle_pts)
say('Making the triangle smaller')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(block_name, pt)

objects = ['brown bowl', 'red block', 'brown block', 'red bowl', 'pink bowl', 'pink block']
# put the red block on the farthest bowl.
farthest_bowl_name = parse_obj_name('the bowl farthest from the red block', f'objects = {get_obj_names()}')
say(f'Putting the red block on the {farthest_bowl_name}')
put_first_on_second('red block', farthest_bowl_name)

""".strip()

# Function generation prompt template
FUNCTION_GENERATION_PROMPT = """
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: result = add_offset(value, offset=10).
def add_offset(value, offset):
    return value + offset

# define function: pt = get_midpoint(pt1, pt2).
def get_midpoint(pt1, pt2):
    return (pt1 + pt2) / 2

# define function: dist = get_distance(pt1, pt2).
def get_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)
""".strip()


# ============================================================================
# Utility Functions
# ============================================================================

def is_chat_model(model_name: str) -> bool:
    """Check if model requires chat completions API"""
    chat_keywords = ['gpt-3.5-turbo', 'gpt-4', 'glm', 'deepseek', 'qwen']
    return any(keyword in model_name.lower() for keyword in chat_keywords)


def exec_safe(code: str, gvars: dict, lvars: dict):
    """Safely execute code"""
    try:
        exec(code, gvars, lvars)
    except Exception as e:
        print(f'Code execution error: {e}')
        print(f'Error code:\n{code}')


def merge_dicts(dicts: List[dict]) -> dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


# ============================================================================
# Function Generator (LMPFGen)
# ============================================================================

class FunctionGenerator:
    """
    Function Generator: Dynamically generate function implementations from signatures
    
    Example:
        >>> fgen = FunctionGenerator(client, model_name)
        >>> add_func = fgen.generate("add(a, b)")
        >>> result = add_func(3, 5)  # returns 8
    """
    
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        prompt_template: str = FUNCTION_GENERATION_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 256
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chat = is_chat_model(model_name)
        
    def generate(self, function_signature: str, verbose: bool = True) -> Callable:
        """
        Generate function implementation from signature
        
        Args:
            function_signature: Function signature like "add(a, b)" or "def add(a, b)"
            verbose: Whether to print generation process
            
        Returns:
            Generated function object
        """
        # Normalize function signature
        if not function_signature.startswith('def '):
            function_signature = f'def {function_signature}'
        if not function_signature.endswith(':'):
            function_signature = f'{function_signature}:'
            
        # Build prompt
        prompt = f"{self.prompt_template}\n\n# define function: {function_signature}"
        
        if verbose:
            print(f'[FunctionGenerator] Generating: {function_signature}')
        
        # Call LLM
        code = self._call_llm(prompt, stop=['# define function'])
        
        # Ensure code contains complete function definition
        if not code.strip().startswith('def '):
            code = f'{function_signature}\n{code}'
        
        if verbose:
            print(f'[FunctionGenerator] Generated code:\n{code}\n')
        
        # Execute code and return function
        namespace = {}
        try:
            exec(code, namespace)
            # Find the defined function
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    return obj
        except Exception as e:
            print(f'[FunctionGenerator] Function generation failed: {e}')
            return None
    
    def extract_functions_from_code(self, code: str) -> Dict[str, Callable]:
        """
        Extract all function definitions from code
        
        Args:
            code: Python code string
            
        Returns:
            Dictionary mapping function names to function objects
        """
        functions = {}
        try:
            # Execute code and extract functions
            namespace = {}
            exec(code, namespace)
            
            # If astunparse is available, extract individual functions more precisely
            if HAS_ASTUNPARSE:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if func_name in namespace and callable(namespace[func_name]):
                            functions[func_name] = namespace[func_name]
            else:
                # Without astunparse, extract all callable objects
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith('_'):
                        functions[name] = obj
        except Exception as e:
            print(f'[FunctionGenerator] Failed to extract functions from code: {e}')
        
        return functions
    
    def _call_llm(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call LLM API"""
        for attempt in range(3):
            try:
                if self.use_chat:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop
                    )
                    return response.choices[0].message.content.strip()
                else:
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=stop
                    )
                    return response.choices[0].text.strip()
            except Exception as e:
                print(f'[FunctionGenerator] API call failed (attempt {attempt+1}/3): {e}')
                if attempt < 2:
                    sleep(2)
                else:
                    raise


# ============================================================================
# Language Model Program (LMP)
# ============================================================================

class LMP:
    """
    Language Model Program (LMP)
    
    Core functionality:
    - Generate Python code from natural language instructions
    - Optionally execute generated code
    - Maintain execution history
    - Support context and examples
    
    Example:
        >>> lmp = LMP(client, model_name, prompt_template)
        >>> code = lmp("pick up the red block")
        >>> lmp("put the blue cylinder on the yellow block", execute=True)
    """
    
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        prompt_template: str = DEFAULT_TASK_PROMPT,
        function_generator: Optional[FunctionGenerator] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        maintain_session: bool = True,
        auto_execute: bool = False,
        verbose: bool = True
    ):
        """
        Initialize LMP
        
        Args:
            client: OpenAI client
            model_name: Model name
            prompt_template: Prompt template (contains example code)
            function_generator: Function generator (for dynamic function creation)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            maintain_session: Whether to maintain session history
            auto_execute: Whether to auto-execute generated code
            verbose: Whether to print detailed information
        """
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.function_generator = function_generator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.maintain_session = maintain_session
        self.auto_execute = auto_execute
        self.verbose = verbose
        self.use_chat = is_chat_model(model_name)
        
        # Execution history
        self.exec_history = []
        
        # Variable spaces
        self.global_vars = {}
        self.custom_functions = {}
        
    def __call__(
        self,
        instruction: str,
        context: str = '',
        execute: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute LMP: generate and optionally execute code
        
        Args:
            instruction: Natural language instruction
            context: Additional context information
            execute: Whether to execute code (None uses auto_execute setting)
            **kwargs: Additional variables passed to execution environment
            
        Returns:
            Dictionary containing:
            - instruction: Original instruction
            - code: Generated code
            - executed: Whether code was executed
            - output: Execution output (if executed)
        """
        # Build complete prompt
        prompt = self._build_prompt(instruction, context)
        
        if self.verbose:
            print(f'\n{"="*60}')
            print(f'[LMP] Instruction: {instruction}')
            if context:
                print(f'[LMP] Context: {context}')
            print(f'[LMP] Generated prompt:\n{prompt}')
        
        # Call LLM to generate code
        code = self._call_llm(prompt)
        
        if self.verbose:
            print(f'[LMP] Generated code:\n{code}')
            print(f'{"="*60}\n')
        
        # Decide whether to execute
        should_execute = execute if execute is not None else self.auto_execute
        
        result = {
            'instruction': instruction,
            'code': code,
            'executed': should_execute,
            'output': None
        }
        
        # Execute code
        if should_execute:
            output = self._execute_code(code, **kwargs)
            result['output'] = output
            
            # Update history
            if self.maintain_session:
                self.exec_history.append({
                    'instruction': instruction,
                    'code': code,
                    'output': output
                })
        
        return result
    
    def _build_prompt(self, instruction: str, context: str = '') -> str:
        """Build complete prompt"""
        # Prepare import statements
        if self.custom_functions:
            imports = f"from utils import {', '.join(self.custom_functions.keys())}"
        else:
            imports = ""
        
        # Replace placeholders in template
        prompt = self.prompt_template.replace('{imports}', imports)
        
        # Add execution history
        if self.maintain_session and self.exec_history:
            history_str = '\n\n'.join([
                f"# {h['instruction']}\n{h['code']}"
                for h in self.exec_history[-3:]  # Keep only last 3
            ])
            prompt += f'\n\n{history_str}'
        
        # Add context
        if context:
            prompt += f'\n\n{context}'
        
        # Add current instruction
        prompt += f'\n# {instruction}'
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API to generate code"""
        for attempt in range(3):
            try:
                if self.use_chat:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{
                            "role": "system",
                            "content": "You are an expert Python code generation assistant for robot control. Given a task description, generate concise, correct Python code to control a robot in a simulated or real environment.    Do not generate explanations or comments—output only executable code for the robot to achieve the described task."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content.strip()
                else:
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stop=['#', "objects = ["]
                    )
                    content = response.choices[0].text.strip()
                
                # Clean code: remove possible markdown code block markers
                content = re.sub(r'^```python\s*\n', '', content)
                content = re.sub(r'\n```\s*$', '', content)
                
                return content
                
            except Exception as e:
                print(f'[LMP] API call failed (attempt {attempt+1}/3): {e}')
                if attempt < 2:
                    sleep(2)
                else:
                    raise
    
    def _execute_code(self, code: str, **kwargs) -> Any:
        """Execute generated code"""
        # Prepare execution environment
        exec_globals = {
            **self.global_vars,
            **self.custom_functions
        }
        exec_locals = kwargs.copy()
        
        # Extract new function definitions
        if self.function_generator:
            new_funcs = self.function_generator.extract_functions_from_code(code)
            exec_globals.update(new_funcs)
            self.custom_functions.update(new_funcs)
        
        try:
            exec_safe(code, exec_globals, exec_locals)
            if self.verbose:
                print(f'[LMP] Code executed successfully')
            return exec_locals
        except Exception as e:
            print(f'[LMP] Code execution failed: {e}')
            return {'error': str(e)}
    
    def register_function(self, name: str, func: Callable):
        """Register custom function for use in generated code"""
        self.custom_functions[name] = func
        
    def set_global_var(self, name: str, value: Any):
        """Set global variable"""
        self.global_vars[name] = value
    
    def clear_history(self):
        """Clear execution history"""
        self.exec_history = []
    
    def get_history(self) -> List[Dict]:
        """Get execution history"""
        return self.exec_history.copy()


# ============================================================================
# Code-as-Policy Main Class
# ============================================================================

class CodeAsPolicy:
    """
    Code-as-Policy Main Class: Simplified Interface
    
    Features:
    - Quick initialization and configuration
    - Manage multiple LMPs
    - Provide convenient code generation interface
    - Support custom functions and tools
    
    Example:
        >>> cap = CodeAsPolicy(provider='deepseek')
        >>> cap.register_action('pick', lambda obj: print(f'Pick up {obj}'))
        >>> cap.generate("pick up the red block")
        >>> cap.execute("put the blue cylinder on the yellow block")
    """
    
    def __init__(
        self,
        provider: str = PROVIDER,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        task_prompt: str = DEFAULT_TASK_PROMPT,
        function_prompt: str = FUNCTION_GENERATION_PROMPT,
        auto_execute: bool = False,
        verbose: bool = True
    ):
        """
        Initialize Code-as-Policy
        
        Args:
            provider: API provider ('zh ipuai', 'deepseek', 'qwen', or 'custom')
            api_key: API key (if None, read from API_CONFIGS)
            base_url: API base URL (if None, read from API_CONFIGS)
            model_name: Model name (if None, read from API_CONFIGS)
            temperature: Sampling temperature
            task_prompt: Task planning prompt template
            function_prompt: Function generation prompt template
            auto_execute: Whether to auto-execute generated code
            verbose: Whether to print detailed information
        """
        # Load configuration
        if provider in API_CONFIGS:
            config = _resolve_api_config(provider)
            api_key = api_key or config['api_key']
            base_url = base_url or config['base_url']
            model_name = model_name or config['model_name']
        else:
            if not all([api_key, base_url, model_name]):
                raise ValueError(f"Unknown provider: {provider}. Please provide api_key, base_url, model_name")
        
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Create OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Create function generator
        self.function_generator = FunctionGenerator(
            client=self.client,
            model_name=model_name,
            prompt_template=function_prompt,
            temperature=temperature
        )
        
        # Create main LMP (task planner)
        self.task_planner = LMP(
            client=self.client,
            model_name=model_name,
            prompt_template=task_prompt,
            function_generator=self.function_generator,
            temperature=temperature,
            auto_execute=auto_execute,
            verbose=verbose
        )
        
        # Register default utility functions
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default utility functions"""
        def say(message):
            """Print message"""
            print(f'[Say]: {message}')
        
        def log(message):
            """Log message"""
            print(f'[Log]: {message}')
        
        self.task_planner.register_function('say', say)
        self.task_planner.register_function('log', log)
    
    def generate(self, instruction: str, context: str = '') -> str:
        """
        Generate code (without execution)
        
        Args:
            instruction: Natural language instruction
            context: Additional context
            
        Returns:
            Generated code string
        """
        result = self.task_planner(instruction, context=context, execute=False)
        return result['code']
    
    def execute(self, instruction: str, context: str = '', **kwargs) -> Dict[str, Any]:
        """
        Generate and execute code
        
        Args:
            instruction: Natural language instruction
            context: Additional context
            **kwargs: Variables passed to execution environment
            
        Returns:
            Dictionary containing code, output, etc.
        """
        return self.task_planner(instruction, context=context, execute=True, **kwargs)
    
    def register_action(self, name: str, function: Callable, description: str = ''):
        """
        Register action function
        
        Args:
            name: Function name
            function: Function object
            description: Function description (optional)
        """
        self.task_planner.register_function(name, function)
        if self.verbose and description:
            print(f'[Register Action] {name}: {description}')
    
    def register_utility(self, name: str, function: Callable, description: str = ''):
        """
        Register utility function
        
        Args:
            name: Function name
            function: Function object
            description: Function description (optional)
        """
        self.task_planner.register_function(name, function)
        if self.verbose and description:
            print(f'[Register Utility] {name}: {description}')
    
    def set_variable(self, name: str, value: Any):
        """Set global variable"""
        self.task_planner.set_global_var(name, value)
    
    def clear_history(self):
        """Clear history"""
        self.task_planner.clear_history()
    
    def get_history(self) -> List[Dict]:
        """Get history"""
        return self.task_planner.get_history()
    
    def set_prompt_template(self, template: str):
        """Update prompt template"""
        self.task_planner.prompt_template = template


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == '__main__':
    # Create Code-as-Policy instance
    cap = CodeAsPolicy(
        provider='zhipuai',  # Options: 'zhipuai', 'qwen', 'deepseek'
        temperature=0.0,
        auto_execute=False,
        verbose=True
    )
    
    # # Register some custom actions
    # def pick(obj_name):
    #     print(f'✓ Pick: {obj_name}')
    
    # def place_on(target):
    #     print(f'✓ Place on: {target}')
    
    # def move_to(position):
    #     print(f'✓ Move to position: {position}')
    
    # def exists(obj_name):
    #     print(f'? Check if exists: {obj_name}')
    #     return True  # Example: always return True
    
    # cap.register_action('pick', pick, 'Pick up object')
    # cap.register_action('place_on', place_on, 'Place object')
    # cap.register_action('move_to', move_to, 'Move to position')
    # cap.register_action('exists', exists, 'Check if object exists')
    
    print('\n' + '='*70)
    print('Code-as-Policy Demo')
    print('='*70)
    
    # Example 1: Generate code only
    print('\nExample 1: Generate code (no execution)')
    print('-'*70)
    code = cap.generate("put the red block into the yellow bowl", "objects = {'red block', 'blue cylinder', 'green block', 'yellow block', 'knife'}")
    print(f'\nGenerated code:\n{code}')
    
    # # Example 2: Generate and execute code
    # print('\n\nExample 2: Generate and execute code')
    # print('-'*70)
    # result = cap.execute("check if yellow object exists, if yes pick it up")
    
    print('\n\nTask completed!')
