import random
import os
import json
import traceback
from alfworld.agents.environment.alfred_thor_env import TASK_TYPES
from alfworld.env.thor_env import ThorEnv


def get_env_paths_solvable(self):
    self.json_file_list = []

    if self.train_eval == "train":
        data_path = os.path.expandvars(self.config['dataset']['data_path'])
    elif self.train_eval == "eval_in_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
    elif self.train_eval == "eval_out_of_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])
    else:
        raise Exception("Invalid split. Must be either train or eval")

    # get task types
    assert len(self.config['env']['task_types']) > 0
    task_types = []
    for tt_id in self.config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_types.append(TASK_TYPES[tt_id])

    for root, dirs, files in os.walk(data_path, topdown=False):
        if 'traj_data.json' in files:
            # Skip movable and slice objects object tasks
            if 'movable' in root or 'Sliced' in root:
                continue

            # File paths
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")

            # Load trajectory file
            with open(json_path, 'r') as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            if not traj_data['task_type'] in task_types:
                continue
            
            ##############################
            # patch: using only the solvable games (134 ood, 140 id)
            ##############################
            # self.json_file_list.append(json_path)

            # # Only add solvable games
            if os.path.exists(game_file_path):
                with open(game_file_path, 'r') as f:
                    gamedata = json.load(f)
            
                if 'solvable' in gamedata and gamedata['solvable']:
                    self.json_file_list.append(json_path)
            ##############################

    print("Overall we have %s games..." % (str(len(self.json_file_list))))
    self.num_games = len(self.json_file_list)

    if self.train_eval == "train":
        num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_train_games]
        self.num_games = len(self.json_file_list)
        print("Training with %d games" % (len(self.json_file_list)))
    else:
        num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_eval_games]
        self.num_games = len(self.json_file_list)
        print("Evaluating with %d games" % (len(self.json_file_list)))

def env_reset_with_idx(self, i=None):
    # set tasks
    batch_size = self.batch_size
    # reset envs
    
    ##############################
    # patch: reset to the task of the given index
    ##############################
    if i is not None:
        assert batch_size == 1
        tasks = [self.json_file_list[i]]
    ##############################
    else:
        if self.train_eval == 'train':
            tasks = random.sample(self.json_file_list, k=batch_size)
        else:
            if len(self.json_file_list)-batch_size > batch_size:
                tasks = [self.json_file_list.pop(random.randrange(len(self.json_file_list))) for _ in range(batch_size)]
            else:
                tasks = random.sample(self.json_file_list, k=batch_size)
                self.get_env_paths()

    for n in range(batch_size):
        self.action_queues[n].put((None, True, tasks[n]))

    obs, dones, infos = self.wait_and_get_info()
    return obs, infos

def modified_oracle_step(self, action_str):
    event = None
    self.feedback = "Nothing happens."

    try:
        cmd = self.parse_command(action_str)

        if cmd['action'] == self.Action.GOTO:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            if recep and recep['num_id'] == self.curr_recep:
                return self.feedback
            self.curr_loc = recep['locs']
            event = self.navigate(self.curr_loc)
            self.curr_recep = recep['num_id']
            self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)

            # feedback conditions
            loc_id = list(self.receptacles.keys()).index(recep['object_id'])
            loc_feedback = "You arrive at loc %s. " % loc_id
            state_feedback = "The {} is {}. ".format(self.curr_recep, "closed" if recep['closed'] else "open") if recep['closed'] is not None else ""
            loc_state_feedback = loc_feedback + state_feedback
            self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
            self.frame_desc = str(self.feedback)

        elif cmd['action'] == self.Action.PICK:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            if obj in self.visible_objects:
                object = self.get_object(obj, self.objects)
                event = self.env.step({'action': "PickupObject",
                                        'objectId': object['object_id'],
                                        'forceAction': True})

                if event.metadata['lastActionSuccess']:
                    self.inventory.append(object['num_id'])
                    self.feedback = "You pick up the %s from the %s." % (obj, tar)

        elif cmd['action'] == self.Action.PUT:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            recep = self.get_object(tar, self.receptacles)
            event = self.env.step({'action': "PutObject",
                                    'objectId': self.env.last_event.metadata['inventoryObjects'][0]['objectId'],
                                    'receptacleObjectId': recep['object_id'],
                                    'forceAction': True})
            if event.metadata['lastActionSuccess']:
                self.inventory.pop()
                self.feedback = "You put the %s %s the %s." % (obj, rel, tar)

        elif cmd['action'] == self.Action.OPEN:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            event = self.env.step({'action': "OpenObject",
                                    'objectId': recep['object_id'],
                                    'forceAction': True})
            self.receptacles[recep['object_id']]['closed'] = False
            self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)
            action_feedback = "You open the %s. The %s is open. " % (target, target)
            self.feedback = action_feedback + self.feedback.replace("On the %s" % target, "In it")
            self.frame_desc = str(self.feedback)

        elif cmd['action'] == self.Action.CLOSE:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            event = self.env.step({'action': "CloseObject",
                                    'objectId': recep['object_id'],
                                    'forceAction': True})
            self.receptacles[recep['object_id']]['closed'] = True
            self.feedback = "You close the %s." % target

        elif cmd['action'] == self.Action.TOGGLE:
            target = cmd['tar']
            obj = self.get_object(target, self.objects)
            event = self.env.step({'action': "ToggleObjectOn",
                                    'objectId': obj['object_id'],
                                    'forceAction': True})
            self.feedback = "You turn on the %s." % target

        elif cmd['action'] == self.Action.HEAT:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            recep = self.get_object(tar, self.receptacles)

            # open the microwave, heat the object, take the object, close the microwave
            events = []
            events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PutObject', 'objectId': obj_id, 'receptacleObjectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': obj_id, 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))

            ##############################
            # patch: Below, ALFWorld originally checks additionally for self.curr_recep == tar. If self.curr_recep != tar, the actions above will still take place but the feedback will be "Nothing happens", and thus the agent cannot understand what really happened. We now disable this check to give a correct feedback.
            ##############################
            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You heat the %s using the %s." % (obj, tar)
            ##############################

        elif cmd['action'] == self.Action.CLEAN:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.env.last_event.metadata['inventoryObjects'][0]
            sink = self.get_obj_cls_from_metadata('BathtubBasin' if "bathtubbasin" in tar else "SinkBasin")
            faucet = self.get_obj_cls_from_metadata('Faucet')

            # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
            events = []
            events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': sink['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': faucet['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': faucet['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))

            ##############################
            # patch: Below, ALFWorld originally checks additionally for self.curr_recep == tar. If self.curr_recep != tar, the actions above will still take place but the feedback will be "Nothing happens", and thus the agent cannot understand what really happened. We now disable this check to give a correct feedback.
            ##############################
            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You clean the %s using the %s." % (obj, tar)
            ##############################

        elif cmd['action'] == self.Action.COOL:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.env.last_event.metadata['inventoryObjects'][0]
            fridge = self.get_obj_cls_from_metadata('Fridge')

            # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
            events = []
            events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))

            ##############################
            # patch: Below, ALFWorld originally checks additionally for self.curr_recep == tar. If self.curr_recep != tar, the actions above will still take place but the feedback will be "Nothing happens", and thus the agent cannot understand what really happened. We now disable this check to give a correct feedback.
            ##############################
            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You cool the %s using the %s." % (obj, tar)
            ##############################

        elif cmd['action'] == self.Action.SLICE:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.get_object(obj, self.objects)
            inventory_objects = self.env.last_event.metadata['inventoryObjects']
            if 'Knife' in inventory_objects[0]['objectType']:
                event = self.env.step({'action': "SliceObject",
                                        'objectId': object['object_id']})
            self.feedback = "You slice %s with the %s" % (obj, tar)

        elif cmd['action'] == self.Action.INVENTORY:
            if len(self.inventory) > 0:
                self.feedback = "You are carrying: a %s" % (self.inventory[0])
            else:
                self.feedback = "You are not carrying anything."

        elif cmd['action'] == self.Action.EXAMINE:
            target = cmd['tar']
            receptacle = self.get_object(target, self.receptacles)
            object = self.get_object(target, self.objects)

            if receptacle:
                self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                self.frame_desc = str(self.feedback)
            elif object:
                self.feedback = self.print_object(object)

        elif cmd['action'] == self.Action.LOOK:
            if self.curr_recep == "nothing":
                self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
            else:
                self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep

    except:
        if self.debug:
            print(traceback.format_exc())

    if event and not event.metadata['lastActionSuccess']:
        self.feedback = "Nothing happens."
        if self.debug:
            print(event.metadata['errorMessage'])

    if self.debug:
        print(self.feedback)
    return self.feedback
   