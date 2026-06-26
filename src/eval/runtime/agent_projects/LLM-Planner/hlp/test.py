import os
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None


if load_dotenv is not None:
    for candidate in (Path(__file__).resolve().parents[4] / '.env', Path.cwd() / '.env'):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)

client = OpenAI(
    api_key=os.getenv("ZHIPUAI_API_KEY", ""),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

prompt = """
Create a high-level plan for completing a household task using the allowed actions and visible objects.

Allowed actions: OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, Navigation

Output only the next plans, without any explanations.

Examples:

Task description: Put two slices of potato into the garbage can.
Completed plans:
Visible objects are diningtable, potatosliced, butterknife, garbagecan, countertop
Next Plans: Navigation countertop, PickupObject butterknife, SliceObject potato, PutObject butterknife countertop, PickupObject potatosliced, Navigation garbagecan, PutObject potatosliced garbagecan, Navigation diningtable, PickupObject butterknife, Navigation potato, SliceObject potato, Navigation 
diningtable, PutObject butterknife diningtable, Navigation diningtable, PickupObject potatosliced, Navigation garbagecan, PutObject potatosliced garbagecan

Task description: Slice the potato using the knife.  Put the knife in the fridge with the potato in the pan.
Completed plans: Navigation diningtable, PickupObject knife, Navigation potato, SliceObject potato, Navigation fridge, OpenObject fridge
Visible objects are knife, potatosliced, fridge, pot
Next Plans: PutObject knife fridge, CloseObject fridge, Navigation diningtable, PickupObject potatosliced, Navigation pot, PutObject potatosliced pot, PickupObject pot, Navigation fridge, OpenObject fridge, PutObject pot fridge, CloseObject fridge

Task description: Put clean potato in the fridge.
Completed plans: Navigation sinkbasin, PickupObject knife, Navigation potato
Visible objects are sink, knife, potatosliced, fridge
Next Plans: OpenObject fridge, SliceObject potato, PutObject knife fridge, PickupObject potatosliced, CloseObject fridge, Navigation sinkbasin, PutObject potatosliced sink, ToggleObjectOn faucet, ToggleObjectOff faucet, PickupObject potatosliced, Navigation fridge, OpenObject fridge, PutObject potatosliced fridge, CloseObject fridge

Task description: Slice a potato, cook a slice, put it in the fridge
Completed plans: Navigation countertop, PickupObject butterknife, Navigation potato, OpenObject fridge, SliceObject potato, CloseObject fridge, Navigation sinkbasin, PutObject butterknife sink, Navigation fridge, OpenObject fridge, PickupObject potatosliced
Visible objects are potatosliced, microwave, fridge, butterknife, sink
Next Plans: CloseObject fridge, Navigation microwave, OpenObject microwave, PutObject potatosliced microwave, CloseObject microwave, ToggleObjectOn microwave, ToggleObjectOff microwave, OpenObject microwave, PickupObject potatosliced, CloseObject microwave, Navigation fridge, OpenObject fridge, PutObject potatosliced fridge, CloseObject fridge

Task description: Put a heated potato slice in the fridge.
Completed plans: Navigation diningtable, PickupObject knife, Navigation potato, SliceObject potato, Navigation microwave, OpenObject microwave, PutObject knife microwave, CloseObject microwave, Navigation diningtable, PickupObject potatosliced, Navigation microwave, OpenObject microwave, PutObject potatosliced microwave, CloseObject microwave, ToggleObjectOn microwave, ToggleObjectOff microwave, OpenObject microwave, PickupObject potatosliced, CloseObject microwave, Navigation fridge
Visible objects are knife, potatosliced, fridge, microwave
Next Plans: OpenObject fridge, PutObject potatosliced fridge, CloseObject fridge

Task description: Put a washed slice of potato in the microwave.
Completed plans: Navigation countertop, PickupObject butterknife, Navigation potato, SliceObject potato, Navigation microwave, OpenObject microwave, PutObject butterknife microwave, CloseObject microwave, Navigation diningtable, PickupObject potatosliced, Navigation sinkbasin, PutObject potatosliced sink, ToggleObjectOn faucet, ToggleObjectOff faucet, PickupObject potatosliced, Navigation microwave
Visible objects are sink, potatosliced, butterknife, microwave
Next Plans: OpenObject microwave, PutObject potatosliced microwave, CloseObject microwave

Task description: Move the pot with a potato slice from the freezer to the sink.
Completed plans: Navigation sinkbasin, PickupObject butterknife, Navigation potato, SliceObject potato
Visible objects are potatosliced, pot, butterknife, sink, countertop
Next Plans: PutObject butterknife countertop, PickupObject potatosliced, Navigation pot, OpenObject fridge, PutObject potatosliced pot, PickupObject pot, CloseObject fridge, Navigation sinkbasin, PutObject pot sink

Task description: Slice a potato, rinse one slice.
Completed plans: Navigation diningtable, PickupObject potato, Navigation countertop, PutObject potato countertop, Navigation countertop, PickupObject 
knife, Navigation potato, SliceObject potato, PutObject knife countertop, PickupObject potatosliced, Navigation sinkbasin, PutObject potatosliced sink, ToggleObjectOn faucet
Visible objects are potatosliced, sink, potato, knife, countertop
Next Plans: ToggleObjectOff faucet, PickupObject potatosliced, Navigation countertop, PutObject potatosliced countertop

Task description: Put potatoes in the basket, slice one of them
Completed plans: Navigation countertop, PickupObject potato, Navigation garbagecan, PutObject potato garbagecan, Navigation fridge, OpenObject fridge, PickupObject potato
Visible objects are potato, garbagecan
Next Plans: CloseObject fridge, Navigation garbagecan, PutObject potato garbagecan, Navigation countertop, PickupObject butterknife, Navigation potato, SliceObject potato, SliceObject potato

Task description: Cook the potato and put it into the recycle bin.
Completed plans: Navigation countertop, PickupObject potato, Navigation microwave
Visible objects are cup, microwave, fridge, garbagecan
Next Plans:
"""

completion = client.chat.completions.create(
    model="glm-4.5-air",
    messages=[
        {"role": "user", "content": prompt}
    ],
    top_p=0.7,
    temperature=0.9
)

print(completion.choices[0].message.content)
