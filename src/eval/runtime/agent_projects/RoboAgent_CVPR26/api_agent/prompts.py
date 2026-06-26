FORMAT_CONSTRAINT_PREFIX = """CRITICAL: You MUST follow the output format EXACTLY as specified. Do NOT add any extra text, explanation, or commentary beyond the required format. Your output will be parsed by a program and any deviation will cause a system failure.\n\n"""

aw_prompt_og = """<image>
Locate {} in the image. If you can find it, output the bounding box in json format (including the object class as "label" of the bounding box); if you cannot find it, output no

IMPORTANT OUTPUT FORMAT RULES:
- If found: output ONLY a json code block in this EXACT format:
```json
[{{"label": "ObjectName 1"}}]
```
- If not found: output ONLY the word "no" with nothing else
- Do NOT output any other text, explanation, or markdown outside the code block"""

aw_prompt_sd = """<image>
This is an egocentric image observed by a robotic household agent. Please describe the {} in the scene."""

aw_prompt_lpe = """Suppose you are a helpful robotic agent in an indoor environment. You are able to perform the following actions:
1. go to <object>
2. open <object>
3. close <object>
The <object> is an object name composed of an object class and an object ID (e.g., Apple 1, DiningTable 2).
Now, your task is to '{}'. Please complete this task by performing one or a series of actions.

CRITICAL OUTPUT FORMAT: You MUST output a list of actions in EXACTLY this format: [action1, action2, action3]
- Each action follows the format: verb ObjectClass ObjectID (e.g., go to Fridge 1, open Cabinet 2)
- The entire output must be enclosed in square brackets [ ]
- Actions must be separated by commas
- Do NOT output ANY other text, explanation, or commentary

Examples:
- [go to Fridge 1]
- [go to Cabinet 2, open Cabinet 2]
- [go to CounterTop 1, go to Fridge 1, open Fridge 1]"""

aw_prompt_lpm = """Suppose you are a helpful robotic agent in an indoor environment. You are able to perform the following actions:
1. go to <object>
2. open <object>
3. close <object>
4. use <object> (which means to turn on the object)
5. take <object> from <object> (which means to grasp something from its receptacle)
6. put <object> to <object> (which means to put something you are holding to a receptacle)
7. cool <object> with <object> (which means to make something you are holding cold with a tool receptacle)
8. heat <object> with <object> (which means to make something you are holding hot with a tool receptacle)
9. clean <object> with <object> (which means to make something you are holding clean with a tool receptacle)
The <object> is an object name composed of an object class and an object ID (e.g., Apple 1, DiningTable 2).
Now, you are holding {}. You are at {}. The environment infomation is: {}
Your task is to '{}'. Please generate a list of actions in the given format to complete this task.

CRITICAL OUTPUT FORMAT: You MUST output a list of actions in EXACTLY this format: [action1, action2, action3]
- Each action follows the exact verb-object format shown above (e.g., go to Fridge 1, take Apple 1 from CounterTop 2)
- The entire output must be enclosed in square brackets [ ]
- Actions must be separated by commas
- Do NOT output ANY other text, explanation, or commentary

Examples:
- [take Cup 1 from CounterTop 2]
- [open Cabinet 1, put Apple 3 to Cabinet 1, close Cabinet 1]
- [go to Fridge 1, open Fridge 1, take Apple 1 from Fridge 1, go to Microwave 1, heat Apple 1 with Microwave 1]"""

aw_prompt_eg = """Suppose you are a helpful robotic agent in an indoor environment. Your task is to find '{}' in the house, based on common house layouts and object placements. Currently, you can observe a list of big objects in the house: {}. Previously, you have tried the following exploration directions: {}. Do not output the directions in this list again since they all failed.

CRITICAL OUTPUT FORMAT: You MUST output EXACTLY ONE exploration direction in the form:
<relation> <object>
where:
- <relation> is ONE of: target, in, on, near
- <object> is an object from the given object list above

RULES:
- Output ONLY the relation and object, NOTHING ELSE
- Do NOT add any explanation, period, or extra words
- Do NOT output a direction that has already been tried

Examples of valid outputs:
- target Apple 1
- in Fridge 1
- on CounterTop 2
- near Microwave 1"""

aw_prompt_es = """<image>
Suppose you are a helpful robotic agent in an indoor environment. Your task is to '{}'. Here is a list of actions you have performed and their corresponding environment feedbacks:
{}
Your current egocentric observation is shown in the image. Please summarize the progress made and analyze the reasons for the failures (if any)."""

aw_prompt_ct = """Suppose you are a helpful robotic agent in an indoor environment. You have the following abilities and you can invoke them by function calling:
1. exploration_guidance(object_information): given the name or the description of an object, output a direction for exploration
2. exploration_planner(): explore the environment according to the exploration direction
3. object_grounding(object_information): given the name or the description of an object, find it in the egocentric view of the robot
4. scene_description(): describe the egocentric observation of the robot
5. manipulation_planner(subtask): given a subtask instruction (a subtask is defined as a part of the task that can be completed within the scene observed from the robot's current egocentric view), complete it by performing atomic actions
6. experience_summarization(subtask): summarize the previous subtask execution experience
7. question_answering(question): answer a question based on the egocentric view of the robot

You need to complete a given task by sequentially generating ability queries. When querying the ability of exploration_guidance, object_grounding, manipulation_planner, experience_summarization, you need to give them a proper input argument. After querying object_grounding or experience_summarization, you will get feedbacks of the grounding results or execution results. At each step, you will be given the history of queries you made and feedbacks you received.

CRITICAL OUTPUT FORMAT: You MUST follow this EXACT format:
Think: <your reasoning process>
Query: 1. <ability_name>(<input_argument>)
2. <ability_name>(<input_argument>)

Or, if you believe the task is completed:
Think: <your reasoning>
Stop

RULES:
- You MUST start with "Think: " followed by your reasoning
- You MUST then have "Query: " followed by numbered ability calls, each on a new line
- Each query MUST be in the EXACT format: ability_name(argument)
- The parentheses () are REQUIRED around the argument
- Do NOT use any other format such as bullet points, dashes, or colons
- Do NOT add any text before "Think:" or after the queries

Examples of valid outputs:
Think: I need to first find the Apple. I should use exploration_guidance to locate it.
Query: 1. exploration_guidance(Apple 1)

Think: I need to find the Apple and then heat it. Let me first get exploration guidance and then ground the object.
Query: 1. exploration_guidance(Apple 1)
2. object_grounding(Apple 1)

Think: The Apple has been found at the CounterTop. Now I need to plan the manipulation to heat it.
Query: 1. manipulation_planner(heat Apple 1 with Microwave 1)

Think: The manipulation failed. Let me summarize the experience and try again.
Query: 1. experience_summarization(heat Apple 1 with Microwave 1)

Think: The task is completed successfully.
Stop

Now, here comes the task:
Task: {}
History:
{}
 """

ebalf_prompt_og = """<image>
Locate {} in the image. If you can find it, output the bounding box in json format (including the object class as "label" of the bounding box); if you cannot find it, output no

IMPORTANT OUTPUT FORMAT RULES:
- If found: output ONLY a json code block in this EXACT format:
```json
[{{"label": "ObjectName 1"}}]
```
- If not found: output ONLY the word "no" with nothing else
- Do NOT output any other text, explanation, or markdown outside the code block"""

ebalf_prompt_sd = """<image>
This is an egocentric image observed by a robotic household agent. Please describe the {} in the scene."""

ebalf_prompt_lpe = """Suppose you are a helpful robotic agent in an indoor environment. You are able to perform the following actions:
1. find a <object>
2. open the <object>
3. close the <object>
The <object> is an object name composed of an object class and an object ID (e.g., Apple 1, DiningTable 2).
Now, your task is to '{}'. Please complete this task by performing one or a series of actions.

CRITICAL OUTPUT FORMAT: You MUST output a list of actions in EXACTLY this format: [action1, action2, action3]
- Each action follows the format shown above (e.g., find a Fridge 1, open the Cabinet 2)
- The entire output must be enclosed in square brackets [ ]
- Actions must be separated by commas
- Do NOT output ANY other text, explanation, or commentary

Examples:
- [find a Cabinet 3]
- [find a Fridge 1, open the Fridge 1]
- [find a CounterTop 1, find a Cabinet 2, open the Cabinet 2]"""

ebalf_prompt_lpm = """Suppose you are a helpful robotic agent in an indoor environment. You are able to perform the following actions:
1. find a <object>
2. open the <object>
3. close the <object>
4. turn on the <object>
5. turn off the <object>
6. pick up the <object>
7. put down the object in hand
8. slice the <object>
The <object> is an object name composed of an object class and an object ID (e.g., Apple 1, DiningTable 2).
Now, you are holding {}. You are at {}. The environment infomation is: {}
Your task is to '{}'. Please generate a list of actions in the given format to complete this task.

Hint:
1. You should always "find" a receptacle before opening it or putting another object to it, unless you are already at that receptacle. You only need to "find" once for a task.
2. If you cannot determine the object ID, simply use 1 as a default choice.

CRITICAL OUTPUT FORMAT: You MUST output a list of actions in EXACTLY this format: [action1, action2, action3]
- Each action follows the exact verb-object format shown above
- The entire output must be enclosed in square brackets [ ]
- Actions must be separated by commas
- Do NOT output ANY other text, explanation, or commentary

Examples:
- [pick up the Cup 1]
- [find a Cabinet 1, open the Cabinet 1, put down the object in hand]
- [find a Fridge 1, open the Fridge 1, pick up the Apple 1]"""

ebalf_prompt_eg = """Suppose you are a helpful robotic agent in an indoor environment. Your task is to find '{}' in the house, based on common house layouts and object placements. Currently, you can observe the following objects in the house: {}. Previously, you have tried the following exploration directions: {}. Do not output the directions in this list again since they all failed.

CRITICAL OUTPUT FORMAT: You MUST output EXACTLY ONE exploration direction in the form:
<relation> <object>
where:
- <relation> is ONE of: target, in, on, near
- <object> is an object from the given object list above

RULES:
- Output ONLY the relation and object, NOTHING ELSE
- Do NOT add any explanation, period, or extra words
- Do NOT output a direction that has already been tried

Examples of valid outputs:
- target Apple 1
- in Fridge 1
- on CounterTop 2
- near Microwave 1"""

ebalf_prompt_es = """<image>
Suppose you are a helpful robotic agent in an indoor environment. Your task is to '{}'. Here is a list of actions you have performed and their corresponding environment feedbacks:
{}
Your current egocentric observation is shown in the image. Please summarize the progress made and analyze the reasons for the failures (if any)."""

ebalf_prompt_ct = """Suppose you are a helpful robotic agent in an indoor environment. You have the following abilities and you can invoke them by function calling:
1. exploration_guidance(object_information): given the name or the description of an object, output a direction for exploration
2. exploration_planner(): explore the environment according to the exploration direction
3. object_grounding(object_information): given the name or the description of an object, find it in the egocentric view of the robot
4. scene_description(): describe the egocentric observation of the robot
5. manipulation_planner(subtask): given a subtask instruction (a subtask is defined as a part of the task that can be completed within the scene observed from the robot's current egocentric view), complete it by performing atomic actions
6. experience_summarization(subtask): summarize the previous subtask execution experience
7. question_answering(question): answer a question based on the egocentric view of the robot

You need to complete a given task by sequentially generating ability queries. When querying the ability of exploration_guidance, object_grounding, manipulation_planner, experience_summarization, you need to give them a proper input argument. After querying object_grounding or experience_summarization, you will get feedbacks of the grounding results or execution results. At each step, you will be given the history of queries you made and feedbacks you received.

CRITICAL OUTPUT FORMAT: You MUST follow this EXACT format:
Think: <your reasoning process>
Query: 1. <ability_name>(<input_argument>)
2. <ability_name>(<input_argument>)

Or, if you believe the task is completed:
Think: <your reasoning>
Stop

RULES:
- You MUST start with "Think: " followed by your reasoning
- You MUST then have "Query: " followed by numbered ability calls, each on a new line
- Each query MUST be in the EXACT format: ability_name(argument)
- The parentheses () are REQUIRED around the argument
- Do NOT use any other format such as bullet points, dashes, or colons
- Do NOT add any text before "Think:" or after the queries

Examples of valid outputs:
Think: I need to first find the Apple. I should use exploration_guidance to locate it.
Query: 1. exploration_guidance(Apple 1)

Think: I need to find the Apple and then heat it. Let me first get exploration guidance and then ground the object.
Query: 1. exploration_guidance(Apple 1)
2. object_grounding(Apple 1)

Think: The Apple has been found at the CounterTop. Now I need to plan the manipulation to heat it.
Query: 1. manipulation_planner(heat Apple 1 with Microwave 1)

Think: The manipulation failed. Let me summarize the experience and try again.
Query: 1. experience_summarization(heat Apple 1 with Microwave 1)

Think: The task is completed successfully.
Stop

Now, here comes the task:
Task: {}
History:
{}
 """


PROMPTS = {
    "alfworld": {
        "prompt_og": aw_prompt_og,
        "prompt_sd": aw_prompt_sd,
        "prompt_lpe": aw_prompt_lpe,
        "prompt_lpm": aw_prompt_lpm,
        "prompt_eg": aw_prompt_eg,
        "prompt_es": aw_prompt_es,
        "prompt_ct": aw_prompt_ct,
    },
    "eb-alfred": {
        "prompt_og": ebalf_prompt_og,
        "prompt_sd": ebalf_prompt_sd,
        "prompt_lpe": ebalf_prompt_lpe,
        "prompt_lpm": ebalf_prompt_lpm,
        "prompt_eg": ebalf_prompt_eg,
        "prompt_es": ebalf_prompt_es,
        "prompt_ct": ebalf_prompt_ct,
    },
}
