'''
Scene Generator - Isaac Sim Runtime Host

This file is the long-running Isaac Sim runtime process for the "scene generator" project.

Responsibilities:
1. Start Isaac Sim, load (or create) the target USD stage, and initialize world basics
    (default prim, dome light, ground plane, and helper cameras).
2. Host simulation-side tool execution that must run inside Isaac Sim Python runtime.
3. Maintain bidirectional pipe communication with the external scene generation process
    (scene_generate.py): receive commands, execute tool logic, and send responses.

Runtime model:
- Process A (this file): persistent Isaac Sim process.
- Process B (scene_generate.py): task planner/generator process.
- Both processes communicate through named pipes (Windows) or Unix sockets (Linux).
'''

import argparse
import json
import os
import queue
import sys
import typing

# Parse arguments before importing SimulationApp
parser = argparse.ArgumentParser(description='Isaac Sim Scene Generation Application')
parser.add_argument(
    '--usd-file',
    type=str,
    default=os.environ.get("SCENE_USD_OUTPUT", ""),
    help='Path of the USD file to load, or create if it does not exist',
)
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
parser.add_argument('--room-name', type=str, default='', help='Current room name to operate in, if applicable')
parser.add_argument(
    '--texture-memory-budget',
    type=float,
    default=0.3,
    help='Texture streaming memory budget as a fraction of GPU memory, e.g. 0.3 for 30%%',
)
parser.add_argument(
    '--dlss-exec-mode',
    type=int,
    choices=(0, 1, 2, 3),
    default=0,
    help='DLSS mode: 0=Performance, 1=Balanced, 2=Quality, 3=Auto',
)
parser.add_argument(
    '--disable-viewport-updates',
    action='store_true',
    help='Disable default viewport updates in headless mode',
)
parser.add_argument(
    '--skip-material-loading',
    action='store_true',
    help='Skip material loading to reduce memory when visual fidelity is not needed',
)
parser.add_argument('--viewport-width', type=int, default=1280, help='Viewport/render-product width')
parser.add_argument('--viewport-height', type=int, default=720, help='Viewport/render-product height')
parser.add_argument(
    '--pipe-id',
    type=str,
    default='',
    help='Unique instance identifier used to derive pipe names for parallel multi-instance runs. '
         'When non-empty, "_{pipe_id}" is appended to the default pipe/pipe names so that '
         'multiple Isaac Sim processes can coexist without pipe-name collisions.',
)

args = parser.parse_args()
if not 0.0 < args.texture_memory_budget <= 1.0:
    parser.error('--texture-memory-budget must be in the range (0.0, 1.0]')

from isaacsim import SimulationApp


def _build_launch_config(headless: bool) -> dict:
    extra_args = [
        f'--/rtx-transient/resourcemanager/texturestreaming/memoryBudget={args.texture_memory_budget}',
        f'--/rtx/post/dlss/execMode={args.dlss_exec_mode}',
    ]
    if args.skip_material_loading:
        extra_args.append('--/app/renderer/skipMaterialLoading=true')

    config = {
        'headless': headless,
        'width': args.viewport_width,
        'height': args.viewport_height,
        'extra_args': extra_args,
    }
    if args.disable_viewport_updates and headless:
        config['disable_viewport_updates'] = True
    elif args.disable_viewport_updates:
        print('Ignoring --disable-viewport-updates because it is only supported in headless mode.')
    return config


def _create_simulation_app():
    if sys.platform == 'win32':
        # Windows config: non-headless mode (preserve original behavior)
        return SimulationApp(launch_config=_build_launch_config(headless=args.headless))

    if sys.platform == 'linux':
        # Linux config
        config = _build_launch_config(headless=True)
        config.update({
            # "width": 1280,
            # "height": 720,
            # "window_width": 1920,
            # "window_height": 1080,
            'hide_ui': False,  # Show the GUI
            # "renderer": "RaytracedLighting",
            'display_options': 3286,  # Set display options to show default grid
        })

        app = SimulationApp(launch_config=config)

        from isaacsim.core.utils.extensions import enable_extension

        # Default Livestream settings
        app.set_setting('/app/window/drawMouse', True)
        if args.disable_viewport_updates:
            print('Skipping livestream extension because viewport updates are disabled.')
        else:
            # Enable Livestream extension
            enable_extension('omni.services.livestream.nvcf')
        return app

    raise OSError(f'Unsupported platform: {sys.platform}')


def _apply_memory_optimization_settings(sim_app: SimulationApp) -> None:
    import carb

    settings = carb.settings.get_settings()
    settings.set_float(
        '/rtx-transient/resourcemanager/texturestreaming/memoryBudget',
        float(args.texture_memory_budget),
    )
    settings.set_int('/rtx/post/dlss/execMode', int(args.dlss_exec_mode))
    if args.skip_material_loading:
        settings.set_bool('/app/renderer/skipMaterialLoading', True)

    print(
        'Applied Isaac Sim memory settings: '
        f'texture_memory_budget={args.texture_memory_budget}, '
        f'dlss_exec_mode={args.dlss_exec_mode}, '
        f'skip_material_loading={args.skip_material_loading}, '
        f'disable_viewport_updates={args.disable_viewport_updates and (args.headless or sys.platform == "linux")}'
    )


simulation_app = _create_simulation_app()
_apply_memory_optimization_settings(simulation_app)

from Tools.tool_implementation_isaac import (
    collect_assets,
    create_room,
    delete_object,
    focus_on_prim,
    get_center_of_surface,
    get_object_bbox,
    get_object_position,
    get_room_context,
    get_size_of_object,
    get_sub_prims,
    modify_orientation,
    place_assets,
    query_floor_space,
    query_surface_status,
    record_scene_snapshot,
    resolve_placement_intent,
    scale_asset_uniform,
    resize_assets_to_dimensions,
    scan_scene,
    set_object_pose,
    spawn_object,
)
from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from omni.isaac.core.utils.stage import is_stage_loading
from pipe_communication import PipeCommunicationServer
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

import carb
import omni.usd


def compute_path_bbox(prim_path: str) -> typing.Tuple[carb.Double3, carb.Double3]:
    """
    Compute Bounding Box using omni.usd.UsdContext.compute_path_world_bounding_box
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.UsdContext.html#omni.usd.UsdContext.compute_path_world_bounding_box

    Args:
        prim_path: A prim path to compute the bounding box.
    Returns:
        A range (i.e. bounding box) as a minimum point and maximum point.
    """
    return omni.usd.get_context().compute_path_world_bounding_box(prim_path)


class IsaacSimAppRunner:
    def __init__(self, sim_app: SimulationApp, parsed_args: argparse.Namespace):
        self.simulation_app = sim_app
        self.args = parsed_args
        self.usd_file_path = parsed_args.usd_file
        self.current_room_name = parsed_args.room_name

        self.usd_context = omni.usd.get_context()
        self.stage = None
        self.my_world = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.pipe_server = None

    @staticmethod
    def _is_windows_absolute_path(path: str) -> bool:
        """Return True if *path* looks like a Windows absolute path (e.g. C:\\...)."""
        if len(path) >= 2 and path[1] == ':':
            return True
        if path.startswith('\\\\'):
            return True
        return False

    def _load_or_create_stage(self):
        # Open the specified USD file
        usd_file_path = self.args.usd_file

        # Guard: on Linux, reject Windows-style absolute paths (e.g. D:\...)
        # which would be silently treated as relative paths, creating
        # broken directory trees like /home/user/C:/some/path/...
        if sys.platform != 'win32' and usd_file_path:
            if self._is_windows_absolute_path(usd_file_path):
                print(
                    f'WARNING: --usd-file "{usd_file_path}" is a Windows absolute path '
                    f'but we are running on Linux.  Ignoring it — please provide a '
                    f'Linux-style path (e.g. /home/user/scene/output.usda).'
                )
                usd_file_path = ''

        if not usd_file_path:
            # Fallback when no --usd-file is provided. Override via
            # SCENE_USD_OUTPUT in the environment for a custom location.
            usd_file_path = os.environ.get("SCENE_USD_OUTPUT")
            if not usd_file_path:
                if sys.platform == 'win32':
                    usd_file_path = os.path.join(
                        os.path.expanduser("~"), "scene_output", "scene.usda"
                    )
                else:
                    usd_file_path = '/tmp/isaac_sim_scene.usda'

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(usd_file_path)), exist_ok=True)

        self.usd_file_path = usd_file_path

        if not os.path.exists(self.usd_file_path):
            print(f'ERROR: USD file not found at: {self.usd_file_path}')
            print('Creating a new empty stage instead...')
            Usd.Stage.CreateNew(self.usd_file_path)
            self.usd_context.open_stage(self.usd_file_path)
        else:
            print(f'Loading USD file from: {self.usd_file_path}')
            try:
                success = self.usd_context.open_stage(self.usd_file_path)
                if not success:
                    print(f'ERROR: Failed to open USD file: {self.usd_file_path}')
                    print('Creating a new empty stage instead...')
                    Usd.Stage.CreateNew(self.usd_file_path)
                    self.usd_context.open_stage(self.usd_file_path)
                else:
                    print('USD file loaded successfully!')
            except Exception as e:
                print(f'ERROR: Exception while opening USD file: {e}')
                raise SystemExit(0)

        # Wait for loading to complete
        print('Waiting for stage to load...')
        while is_stage_loading():
            self.simulation_app.update()

        self.stage = self.usd_context.get_stage()

    def _setup_world_lights_and_cameras(self):
        default_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path('/World'))
        self.stage.SetDefaultPrim(default_prim.GetPrim())

        # Add lighting
        dome_light = UsdLux.DomeLight.Define(self.stage, '/World/DomeLight')
        dome_light.CreateIntensityAttr().Set(2000.0)  # Set intensity
        dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        # Add ground plane if it does not exist
        if not self.stage.GetPrimAtPath('/World/GroundPlane'):
            GroundPlane(prim_path='/World/GroundPlane', z_position=0)

        # Add three cameras at different angles
        # 1. Top View Camera
        if not self.stage.GetPrimAtPath('/World/TopCamera'):
            top_camera = UsdGeom.Camera.Define(self.stage, '/World/TopCamera')
            top_camera_xform = UsdGeom.Xformable(top_camera.GetPrim())
            top_camera_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 20))  # 20 meters above on Z axis
            top_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))  # Looking down
            top_camera.GetHorizontalApertureAttr().Set(20)
            top_camera.GetVerticalApertureAttr().Set(20)

        # 2. Side View Camera
        if not self.stage.GetPrimAtPath('/World/SideCamera'):
            side_camera = UsdGeom.Camera.Define(self.stage, '/World/SideCamera')
            side_camera_xform = UsdGeom.Xformable(side_camera.GetPrim())
            side_camera_xform.AddTranslateOp().Set(Gf.Vec3d(20, 0, 0))  # 20 meters to the right on X axis
            side_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, -90, 0))  # Looking left
            side_camera.GetHorizontalApertureAttr().Set(20)
            side_camera.GetVerticalApertureAttr().Set(20)

        # 3. Main View Camera
        if not self.stage.GetPrimAtPath('/World/MainCamera'):
            main_camera = UsdGeom.Camera.Define(self.stage, '/World/MainCamera')
            main_camera_xform = UsdGeom.Xformable(main_camera.GetPrim())
            main_camera_xform.AddTranslateOp().Set(Gf.Vec3d(0, 20, 0))  # 20 meters back on Y axis
            main_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
            main_camera.GetHorizontalApertureAttr().Set(20)
            main_camera.GetVerticalApertureAttr().Set(20)

        print('saving stage')
        # self.stage.GetRootLayer().Save()

    def _setup_runtime(self):
        self.my_world = World(stage_units_in_meters=1.0)

        # Scale the current viewport camera
        camera_prim = self.stage.GetPrimAtPath('/OmniverseKit_Persp')
        camera = UsdGeom.Camera(camera_prim)
        camera.GetHorizontalApertureAttr().Set(20)
        camera.GetVerticalApertureAttr().Set(20)

        # Start the named-pipe listeners (daemon threads managed by the communication server)
        self.pipe_server = PipeCommunicationServer(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            pipe_id=self.args.pipe_id,
        )
        self.pipe_server.start()

    def _save_stage(self) -> bool:
        """Persist the currently opened USD stage to disk."""
        if self.stage is None:
            print('Cannot save stage: no stage is currently loaded.')
            return False

        try:
            self.stage.Save()
            print(f'Stage saved successfully: {self.usd_file_path}')
            return True
        except Exception as e:
            print(f'Failed to save stage: {e}')
            return False

    def _room_index_payload(self) -> dict:
        """Build a compact room/object index for downstream EvalScenario metadata."""
        room_index = {'rooms': {}}
        if self.stage is None:
            return room_index

        rooms_root = self.stage.GetPrimAtPath('/World/rooms')
        if not rooms_root or not rooms_root.IsValid():
            return room_index

        for room_prim in rooms_root.GetChildren():
            if room_prim is None or not room_prim.IsValid():
                continue

            room_name = room_prim.GetName()
            objects = [
                child.GetName()
                for child in room_prim.GetChildren()
                if child is not None and child.IsValid()
            ]
            room_index['rooms'][room_name] = {
                'room_name': room_name,
                'prim_path': str(room_prim.GetPath()),
                'objects': objects,
                'object_count': len(objects),
            }
        return room_index

    def handle_command(self, cmd):
        '''handle commands from agent'''
        if cmd.startswith('collect_assets'):
            _, assets_str = cmd.split(',', 1)
            # Parse format: asset1:count1,asset2:count2,...
            assets_dict = {}
            if assets_str:
                items = assets_str.split(',')
                for item in items:
                    if ':' in item:
                        asset, count_str = item.split(':', 1)
                        try:
                            count = int(count_str)
                            assets_dict[asset] = count
                        except ValueError:
                            print(f'Invalid count for asset {asset}: {count_str}')
                            assets_dict[asset] = 1  # Default: collect 1
                    else:
                        # No colon means default count of 1
                        assets_dict[item] = 1
            result = collect_assets(assets_dict)
            if result:
                self.output_queue.put('True')

        elif cmd.startswith('scale_asset_uniform'):
            _, obj_name, raw_scale_factor = cmd.split(',', 2)
            path = f'/World/rooms/{self.current_room_name}/{obj_name}'
            result = scale_asset_uniform(path, float(raw_scale_factor))
            self.output_queue.put(str(result))

        elif cmd.startswith('resize_assets'):
            _, obj_name, _length, _width, _height = cmd.split(',', 4)

            path = f'/World/rooms/{self.current_room_name}/{obj_name}'
            target_length = float(_length)
            target_width = float(_width)
            target_height = float(_height)
            result = resize_assets_to_dimensions(
                path,
                (target_length, target_width, target_height),
            )
            if result:
                self.output_queue.put('True')
            else:
                self.output_queue.put('False')

        elif cmd.startswith('place_assets'):
            _, path, ox, oy, oz, tx, ty, tz = cmd.split(',', 7)
            original_pos = {'x': float(ox), 'y': float(oy), 'z': float(oz)}
            target_pos = {'x': float(tx), 'y': float(ty), 'z': float(tz)}
            result = place_assets(path, original_pos, target_pos)
            if result:
                self.output_queue.put('True')

        elif cmd.startswith('get_center_of_surface'):
            _, path, surface = cmd.split(',', 2)
            center = get_center_of_surface(path, surface)
            if center:
                self.output_queue.put(f'True,{center[0]},{center[1]},{center[2]}')
            else:
                self.output_queue.put('False,0,0,0')

        elif cmd.startswith('get_size_of_object'):
            _, path = cmd.split(',', 1)
            size = get_size_of_object(path)
            if size:
                self.output_queue.put(f'True,{size[0]},{size[1]},{size[2]}')
            else:
                self.output_queue.put('False,0,0,0')

        elif cmd.startswith('get_sub_prims'):
            _, path = cmd.split(',', 1)
            children = get_sub_prims(path)
            children = ','.join(x for x in children)
            self.output_queue.put(children)

        elif cmd.startswith('record_scene_snapshot'):
            # Call the tool implementation function
            result = record_scene_snapshot()
            # Send the result back to the agent
            self.output_queue.put(result)

        elif cmd.startswith('focus_on_prim'):
            _, camera_path, prim_path = cmd.split(',', 2)
            result = focus_on_prim(camera_path, prim_path)
            if result:
                self.output_queue.put('True')
            else:
                self.output_queue.put('False')

        elif cmd.startswith('modify_orientation'):
            _, path, rx, ry, rz = cmd.split(',', 4)
            rotation = {'x': float(rx), 'y': float(ry), 'z': float(rz)}
            result = modify_orientation(path, rotation)
            if result:
                self.output_queue.put('True')
            else:
                self.output_queue.put('False')

        elif cmd.startswith('create_room'):
            # Command format: create_room,room1,room2,...,house_length,connectivity
            # connectivity format: [(room1,room2),...]
            import ast

            payload = cmd[len('create_room,'):]
            rooms, house_length, connectivity = ast.literal_eval('(' + payload + ')')
            # Replace spaces in room names with underscores
            rooms = [r.replace(' ', '_').replace('-', '_') for r in rooms]
            house_length = int(house_length)
            result = create_room(rooms, house_length, connectivity)
            if result is None:
                boundaries, connections, room_rects = [], {}, []
            else:
                boundaries, connections, room_rects = result
            print('boundaries from create_room:', boundaries)
            print('connections from create_room:', connections)
            if boundaries:
                # Convert boundaries to string representation
                boundary_strs = []
                for room in boundaries:
                    room_strs = []
                    for line in room:
                        line_str = f'(({line[0][0]},{line[0][1]}),({line[1][0]},{line[1][1]}))'
                        room_strs.append(line_str)
                    boundary_strs.append('[' + ','.join(room_strs) + ']')
                boundaries_str = '[' + ','.join(boundary_strs) + ']'

                # Convert connections to string representation
                connection_strs = []
                for conn, line in connections.items():
                    line_strs = []
                    for l in line:
                        line_strs.append(f'(({l[0][0]},{l[0][1]}),({l[1][0]},{l[1][1]}))')
                    connection_str = f'"{conn}":[' + ','.join(line_strs) + ']'
                    connection_strs.append(connection_str)
                connections_str = '{' + ','.join(connection_strs) + '}'

                # room_rects: each room is a list of ((x1,y1),(x2,y2)) sub-rectangles
                room_rect_strs = []
                for room in room_rects:
                    rect_strs = []
                    for rect in room:
                        rect_strs.append(f'(({rect[0][0]},{rect[0][1]}),({rect[1][0]},{rect[1][1]}))')
                    room_rect_strs.append('[' + ','.join(rect_strs) + ']')
                room_rects_str = '[' + ','.join(room_rect_strs) + ']'

                self.output_queue.put(f'{boundaries_str}|{connections_str}|{room_rects_str}')
            else:
                self.output_queue.put('False,[],{},[]')

        elif cmd.startswith('scan_scene'):
            # scan_scene() takes no arguments
            scene_objects = scan_scene(self.current_room_name)
            # Convert list of dicts to string representation
            if scene_objects:
                # Format: [{'name': str, 'bbox': {'min': [x,y,z], 'max': [x,y,z]}, 'position': [x,y,z]}, ...]
                obj_dicts = []
                print(scene_objects)
                for obj in scene_objects:
                    bbox_min = obj['bbox']['min']
                    bbox_max = obj['bbox']['max']
                    position = obj['position']
                    obj_dict = {
                        'name': obj['name'],
                        'bbox': {
                            'min': [round(bbox_min[0], 2), round(bbox_min[1], 2), round(bbox_min[2], 2)],
                            'max': [round(bbox_max[0], 2), round(bbox_max[1], 2), round(bbox_max[2], 2)],
                        },
                        'position': None
                        if position is None
                        else [round(position[0], 2), round(position[1], 2), round(position[2], 2)],
                    }
                    obj_dicts.append(obj_dict)
                output_str = str(obj_dicts)
                self.output_queue.put(f'True,{len(scene_objects)},{output_str}')
            else:
                self.output_queue.put('True,0,[]')

        elif cmd.startswith('get_room_context'):
            # get_room_context() takes no arguments, uses current_room_name
            result = get_room_context(self.current_room_name)
            output_str = json.dumps(result, ensure_ascii=False)
            self.output_queue.put(output_str)

        elif cmd.startswith('query_floor_space'):
            # Format: query_floor_space or query_floor_space,near_wall=wall_0 or query_floor_space,region=center
            payload = cmd[len('query_floor_space'):].strip(',')
            near_wall = None
            region = None
            if payload:
                for part in payload.split(','):
                    if part.startswith('near_wall='):
                        near_wall = part[len('near_wall='):]
                    elif part.startswith('region='):
                        region = part[len('region='):]
            result = query_floor_space(self.current_room_name, near_wall=near_wall, region=region)
            output_str = json.dumps(result, ensure_ascii=False)
            self.output_queue.put(output_str)

        elif cmd.startswith('query_surface_status'):
            # Format: query_surface_status,support_object,surface
            parts = cmd.split(',')
            if len(parts) < 2:
                self.output_queue.put(json.dumps({'error': 'missing support_object argument'}))
            else:
                support_object = parts[1]
                surface = parts[2] if len(parts) >= 3 else 'up'
                result = query_surface_status(self.current_room_name, support_object, surface)
                output_str = json.dumps(result, ensure_ascii=False)
                self.output_queue.put(output_str)

        elif cmd.startswith('get_object_bbox'):
            # This function returns bbox; it is converted to size at the Agent Tool level
            _, name = cmd.split(',', 1)
            bbox_result = get_object_bbox(name, self.current_room_name)
            if bbox_result is not None:
                min_range, max_range = bbox_result
                self.output_queue.put(
                    f'True,{min_range[0]},{min_range[1]},{min_range[2]},{max_range[0]},{max_range[1]},{max_range[2]}'
                )
            else:
                self.output_queue.put('False,0,0,0,0,0,0')

        elif cmd.startswith('spawn_object'):
            # Format: spawn_object,object_name,usd_path,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z
            print('spawn command received:', cmd)
            _, object_name, usd_path, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = cmd.split(',', 8)
            position = (float(pos_x), float(pos_y), float(pos_z))
            rotation = (float(rot_x), float(rot_y), float(rot_z))
            obj_name = spawn_object(object_name, usd_path, self.current_room_name, position, rotation)
            print('spawned object name:', obj_name)
            if obj_name:
                self.output_queue.put(f'{obj_name}')
            else:
                self.output_queue.put(None)

        elif cmd.startswith('set_object_pose'):
            # Format: set_object_pose,path,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z
            # Use "None" for optional parameters
            parts = cmd.split(',')
            if len(parts) != 8:  # command + 7 parameters
                print(f'Invalid set_object_pose command format. Expected 8 parts, got {len(parts)}')
                self.output_queue.put('False')
            else:
                _, object_name, pos_x_str, pos_y_str, pos_z_str, rot_x_str, rot_y_str, rot_z_str = parts
                path = f'/World/rooms/{self.current_room_name}/{object_name}'

                # Parse position (optional)
                if (
                    pos_x_str.lower() == 'none'
                    or pos_y_str.lower() == 'none'
                    or pos_z_str.lower() == 'none'
                ):
                    position = None
                else:
                    try:
                        position = (float(pos_x_str), float(pos_y_str), float(pos_z_str))
                    except ValueError:
                        print(f'Invalid position values: {pos_x_str},{pos_y_str},{pos_z_str}')
                        position = None

                # Parse rotation (optional)
                if (
                    rot_x_str.lower() == 'none'
                    or rot_y_str.lower() == 'none'
                    or rot_z_str.lower() == 'none'
                ):
                    rotation = None
                else:
                    try:
                        rotation = (float(rot_x_str), float(rot_y_str), float(rot_z_str))
                    except ValueError:
                        print(f'Invalid rotation values: {rot_x_str},{rot_y_str},{rot_z_str}')
                        rotation = None

                result = set_object_pose(path, position, rotation)
                if result:
                    self.output_queue.put('True')
                else:
                    self.output_queue.put('False')

        elif cmd.startswith('get_object_position'):
            _, path = cmd.split(',', 1)
            position = get_object_position(path)
            if position is not None:
                self.output_queue.put(f'True,{position[0]},{position[1]},{position[2]}')
            else:
                self.output_queue.put('False,0,0,0')

        elif cmd.startswith('resolve_placement_intent'):
            # Format: resolve_placement_intent,{intent_dict_as_string}
            _, intent_str = cmd.split(',', 1)
            try:
                # Parse intent dictionary (using eval like create_room does for connectivity)
                intent = eval(intent_str)
                if not isinstance(intent, dict):
                    print(f'Error: intent must be a dictionary, got {type(intent)}')
                    self.output_queue.put(
                        str(
                            {
                                'success': False,
                                'position': None,
                                'rotation': None,
                                'out_of_bounds': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                'message': 'intent not dict',
                            }
                        )
                    )
                else:
                    # Convert object_name to prim_path if present
                    if 'object_name' in intent:
                        object_name = intent['object_name']
                        intent['prim_path'] = f'/World/rooms/{self.current_room_name}/{object_name}'
                        # Remove object_name field to avoid confusion
                        del intent['object_name']

                    # Also handle support_object if present (for surface placement)
                    if 'support_object' in intent and not intent['support_object'].startswith('/'):
                        support_obj = intent['support_object']
                        intent['support_object'] = f'/World/rooms/{self.current_room_name}/{support_obj}'

                    if isinstance(intent.get('semantic_location'), dict):
                        semantic_location = intent['semantic_location']
                        ref_obj = semantic_location.get('reference_object') or semantic_location.get('relative_to')
                        if isinstance(ref_obj, str) and not ref_obj.startswith('/'):
                            ref_path = f'/World/rooms/{self.current_room_name}/{ref_obj}'
                            prim = self.stage.GetPrimAtPath(ref_path)
                            if prim.IsValid():
                                if 'reference_object' in semantic_location:
                                    semantic_location['reference_object'] = ref_path
                                if 'relative_to' in semantic_location:
                                    semantic_location['relative_to'] = ref_path

                    # Handle reference objects in direction tuples (for floor placement)
                    for direction in ['x_direction', 'y_direction', 'z_direction']:
                        if direction in intent:
                            ref_obj, distance = intent[direction]
                            if not ref_obj.startswith('/'):
                                ref_path = f'/World/rooms/{self.current_room_name}/{ref_obj}'
                                prim = self.stage.GetPrimAtPath(ref_path)
                                if prim.IsValid():
                                    intent[direction] = (ref_path, distance)
                                else:
                                    intent[direction] = (f'/World/{ref_obj}', distance)
                    print('Parsed intent for placement:', intent)
                    result = resolve_placement_intent(intent)
                    print('Result from resolve_placement_intent:', result)
                    if isinstance(result, dict):
                        self.output_queue.put(str(result))
                    elif isinstance(result, tuple) and len(result) == 3:
                        self.output_queue.put(
                            str(
                                {
                                    'success': True,
                                    'position': tuple(result),
                                    'rotation': None,
                                    'out_of_bounds': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                    'message': '',
                                }
                            )
                        )
                    else:
                        self.output_queue.put(
                            str(
                                {
                                    'success': False,
                                    'position': None,
                                    'rotation': None,
                                    'out_of_bounds': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                    'message': 'resolution failed',
                                }
                            )
                        )
            except Exception as e:
                print(f'Error parsing intent or resolving placement: {str(e)}')
                self.output_queue.put(
                    str(
                        {
                            'success': False,
                            'position': None,
                            'rotation': None,
                            'out_of_bounds': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                            'message': str(e),
                        }
                    )
                )

        elif cmd.startswith('select_room'):
            _, room_name = cmd.split(',', 1)
            # Check if the room exists
            path = '/World/rooms' if room_name == '' else f'/World/rooms/{room_name}'
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f'Room {room_name} does not exist at path {path}')
                self.output_queue.put('False')
                return
            self.current_room_name = room_name
            print(f'Current room set to: {self.current_room_name}')
            self.output_queue.put('True')

        elif cmd == 'save_stage':
            saved = self._save_stage()
            self.output_queue.put('True' if saved else 'False')

        elif cmd.startswith('delete_object'):
            _, object_name = cmd.split(',', 1)
            result = delete_object(object_name, self.current_room_name)
            self.output_queue.put('True' if result else 'False')

        elif cmd == 'get_room_index':
            self.output_queue.put(json.dumps(self._room_index_payload(), ensure_ascii=False))

        else:
            print(f'Unknown command received: {cmd}')
            self.output_queue.put('False')

    def run(self):
        self._load_or_create_stage()
        self._setup_world_lights_and_cameras()
        self._setup_runtime()

        while self.simulation_app.is_running():
            try:
                self.my_world.step(render=True)
                cmd = self.input_queue.get_nowait()
                print(f'handling command:{cmd}')
            except queue.Empty:
                continue

            if cmd == 'quit':
                print('Received quit command, exiting.')
                self._save_stage()
                break

            self.handle_command(cmd)


def main():
    app_runner = IsaacSimAppRunner(sim_app=simulation_app, parsed_args=args)
    app_runner.run()


if __name__ == '__main__':
    main()
