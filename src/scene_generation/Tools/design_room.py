import random
import math
import itertools

# Global variable: records already occupied areas
captured_grims = []

# Global variable: records selection statistics and cached data per area (for UCB algorithm)
# area_stats structure: key=area index, value={
#   'selections': int,      # number of selections
#   'total_reward': float,  # accumulated reward
#   'area': float,          # current area (cached)
#   'complexity': int,      # current complexity (cached)
#   'last_reward': float    # last computed reward value
# }
area_stats = {}
global_room_side_length = 0

def design_room(room_number:int, room_side_length:int):
    """
    generate wanted number of rooms within the given side length

    Args:
        room_number (int): The identifier for the room to generate.
        room_side_length (int): The side length of the whole room.

    Returns:
        list: A list containing all the generated rooms.
    """
    global captured_grims, area_stats, global_room_side_length
    # Reset global variables
    captured_grims = []
    area_stats = {}
    global_room_side_length = room_side_length

    if room_side_length <= 0:
        print("Room side length must be greater than 0.")
        return []
    # Area representation: a list of diagonals, each diagonal is a rectangle
    produced_area = []
    # Randomly select room_number cells as starting cells for rooms
    # Each cell is 1x1, coordinates from (0,0) to (room_side_length-1, room_side_length-1)
    if room_number >= 1:
        # Generate all possible cell coordinates
        all_cells = [(x, y) for x in range(room_side_length-1) for y in range(room_side_length-1)]

        # Randomly select room_number non-repeating cells
        if len(all_cells) >= room_number:
            selected_cells = random.sample(all_cells, room_number)
            for cell in selected_cells:
                area = [(cell, (cell[0]+1, cell[1]+1))]
                produced_area.append(area)
                captured_grims.append(area)

        else:
            # Not enough cells to select from
            print("Not enough cells to select from.")
            return []
        print(f"Selected starting cells: {selected_cells}")
        # TODO: perform subsequent processing based on selected starting cells
    else:
        print("Room number must be greater than 0.")
        return []

    # Initialize area cache
    update_area_cache(produced_area)
    # Iteratively fold areas until no more folding is possible
    can_fold = True
    while can_fold:
        # Check if expansion can continue; calculate total area, stop if limit reached
        total_area = sum(area_stats[i]['area'] for i in range(len(produced_area)))
        if total_area >= room_side_length * room_side_length:
            break
        # Select an area to fold
        area_index = choose_area_to_fold(produced_area)
        # If no expandable area, stop the loop
        if area_index == -1:
            print("All areas can no longer expand, stopping expansion")
            break
        # TODO: area_to_fold changes as produced_area updates, causing it to always equal new_area in comparison
        area_to_fold = produced_area[area_index][:]
        # Perform folding
        copy_area_to_fold = area_to_fold[:]
        new_area = fold_area(copy_area_to_fold, area_index)
        # Check if there is any change
        if new_area != area_to_fold:
            produced_area[area_index] = new_area
            can_fold = True
    boundaries = merge_subarea()
    all_connection = connectivity_record(boundaries)
    room_rects = [list(room) for room in captured_grims]
    return boundaries, all_connection, room_rects

def fold_area(area:list, area_index: int = None):
    """
    Starting from a cell, iteratively perform folding to achieve rectangular expansion.
    Folding a rectangle yields a rectangle; folding a sub-rectangle on the boundary produces jagged edges.
    Diagonals represent rectangular areas: the new rectangle from folding the whole area needs its diagonal updated,
    while jagged boundaries from sub-rectangles are represented with new diagonals, adding a new diagonal to the area representation.
    Also ensures each edge of every area is at least 1m long.

    Args:
        area (list): The area to fold.
        area_index (int, optional): Index of the area in the global list, used for cache updates.

    Returns:
        list: The folded area.
    """
    global captured_grims, area_stats
    # Iterate over each sub-area in area, attempt to fold
    for i, target_area in enumerate(area):
        # 1. First try whole-area folding
        # Determine fold direction priority based on aspect ratio
        (x1, y1), (x2, y2) = target_area
        width = x2 - x1
        height = y2 - y1

        # Calculate aspect ratio to decide priority direction
        # If width is relatively smaller than height, prioritize left-right folding to increase width
        # If height is relatively smaller than width, prioritize up-down folding to increase height
        # Threshold set to 0.8 (adjustable)
        if width < height * 0.8:
            # Width is narrow, prioritize left-right folding
            directions = ['left', 'right', 'up', 'down']
        elif height < width * 0.8:
            # Height is narrow, prioritize up-down folding
            directions = ['up', 'down', 'left', 'right']
        else:
            # Aspect ratio is close, random order
            directions = ['up', 'down', 'left', 'right']
            random.shuffle(directions)
        for direction in directions:
            if is_valid_fold(target_area, direction, exclude_region=area):
                if direction == 'up':
                    # Fold upward
                    new_area = (target_area[0], (target_area[1][0], 2*target_area[1][1] - target_area[0][1]))
                    # Update the corresponding area in captured_grims
                    captured_grims[area_index][i] = new_area
                    area[i] = new_area
                    # Update cache
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area
                elif direction == 'down':
                    # Fold downward
                    new_area = ((target_area[0][0], 2*target_area[0][1] - target_area[1][1]), target_area[1])
                    captured_grims[area_index][i] = new_area
                    area[i] = new_area
                    # Update cache
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area
                elif direction == 'left':
                    # Fold left
                    new_area = ((2*target_area[0][0] - target_area[1][0], target_area[0][1]), target_area[1])
                    captured_grims[area_index][i] = new_area
                    area[i] = new_area
                    # Update cache
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area
                elif direction == 'right':
                    # Fold right
                    new_area = (target_area[0], (2*target_area[1][0] - target_area[0][0], target_area[1][1]))
                    captured_grims[area_index][i] = new_area
                    area[i] = new_area
                    # Update cache
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area

        # 2. If none of the four directions allow whole-area folding, check if it is the minimum 1x1 unit
        (x1, y1), (x2, y2) = target_area
        if x2 - x1 <= 1 and y2 - y1 <= 1:
            # Current sub-area is already minimum unit, cannot fold further, skip this sub-area
            continue

        # 3. Try folding using the four complete boundaries
        whole_boundaries = extract_whole_boundary(target_area)
        if whole_boundaries:  # Ensure there are boundaries
            # Phase 1: Prioritize trying each boundary folding in its own direction
            for boundary_area, boundary_direction in whole_boundaries:
                # Try the boundary's own direction
                if is_valid_fold(boundary_area, boundary_direction, exclude_region=area):
                    # Found a valid fold direction, calculate new area
                    if boundary_direction == 'up':
                        # Fold upward: only includes the newly generated area (above boundary_area)
                        new_area = ((boundary_area[0][0], boundary_area[1][1]),
                                    (boundary_area[1][0], 2*boundary_area[1][1] - boundary_area[0][1]))
                    elif boundary_direction == 'down':
                        # Fold downward: only includes the newly generated area (below boundary_area)
                        new_area = ((boundary_area[0][0], 2*boundary_area[0][1] - boundary_area[1][1]),
                                    (boundary_area[1][0], boundary_area[0][1]))
                    elif boundary_direction == 'left':
                        # Fold left: only includes the newly generated area (left of boundary_area)
                        new_area = ((2*boundary_area[0][0] - boundary_area[1][0], boundary_area[0][1]),
                                    (boundary_area[0][0], boundary_area[1][1]))
                    elif boundary_direction == 'right':
                        # Fold right: only includes the newly generated area (right of boundary_area)
                        new_area = ((boundary_area[1][0], boundary_area[0][1]),
                                    (2*boundary_area[1][0] - boundary_area[0][0], boundary_area[1][1]))

                    # Add new area
                    area.append(new_area)
                    # Update the corresponding area in captured_grims
                    try:
                        captured_grims[area_index].append(new_area)
                    except ValueError:
                        # If not found, add new area directly (theoretically should not happen)
                        captured_grims.append([new_area])

                    # Update cache
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area

            # Phase 2: If all four boundaries fail in their own directions, try folding in other directions
            for boundary_area, boundary_direction in whole_boundaries:
                # Opposite direction mapping: the opposite of the boundary direction is the forbidden fold direction
                opposite_direction = {
                    'up': 'down',
                    'down': 'up',
                    'left': 'right',
                    'right': 'left'
                }
                forbidden_direction = opposite_direction[boundary_direction]
                # Build fold direction list: exclude forbidden direction and the boundary's own direction (already tried in Phase 1)
                other_directions = [d for d in ['up', 'down', 'left', 'right']
                                  if d != forbidden_direction and d != boundary_direction]
                random.shuffle(other_directions)  # Randomize the order of other directions

                for fold_direction in other_directions:
                    if is_valid_fold(boundary_area, fold_direction, exclude_region=area):
                        # Found a valid fold direction, calculate new area
                        if fold_direction == 'up':
                            # Fold upward: only includes the newly generated area (above boundary_area)
                            new_area = ((boundary_area[0][0], boundary_area[1][1]),
                                        (boundary_area[1][0], 2*boundary_area[1][1] - boundary_area[0][1]))
                        elif fold_direction == 'down':
                            # Fold downward: only includes the newly generated area (below boundary_area)
                            new_area = ((boundary_area[0][0], 2*boundary_area[0][1] - boundary_area[1][1]),
                                        (boundary_area[1][0], boundary_area[0][1]))
                        elif fold_direction == 'left':
                            # Fold left: only includes the newly generated area (left of boundary_area)
                            new_area = ((2*boundary_area[0][0] - boundary_area[1][0], boundary_area[0][1]),
                                        (boundary_area[0][0], boundary_area[1][1]))
                        elif fold_direction == 'right':
                            # Fold right: only includes the newly generated area (right of boundary_area)
                            new_area = ((boundary_area[1][0], boundary_area[0][1]),
                                        (2*boundary_area[1][0] - boundary_area[0][0], boundary_area[1][1]))

                        # Add new area
                        area.append(new_area)
                        # Update the corresponding area in captured_grims
                        try:
                            captured_grims[area_index].append(new_area)
                        except ValueError:
                            # If not found, add new area directly (theoretically should not happen)
                            captured_grims.append([new_area])

                        # Update cache
                        if area_index is not None:
                            update_single_area_cache(area_index, area)
                        return area

        # 4. If all four complete boundaries cannot fold, try random boundary sub-rectangles
        max_attempts = 100
        attempt = 0
        while attempt < max_attempts:
            # Extract boundary sub-rectangle candidates (returns multiple candidates)
            candidates = extract_boundary_area(target_area)
            if not candidates:
                attempt += 1
                continue

            # First try each candidate's own direction
            for boundary_area, boundary_direction in candidates:
                if is_valid_fold(boundary_area, boundary_direction, exclude_region=area):
                    if boundary_direction == 'up':
                        new_area = ((boundary_area[0][0], boundary_area[1][1]),
                                    (boundary_area[1][0], 2*boundary_area[1][1] - boundary_area[0][1]))
                    elif boundary_direction == 'down':
                        new_area = ((boundary_area[0][0], 2*boundary_area[0][1] - boundary_area[1][1]),
                                    (boundary_area[1][0], boundary_area[0][1]))
                    elif boundary_direction == 'left':
                        new_area = ((2*boundary_area[0][0] - boundary_area[1][0], boundary_area[0][1]),
                                    (boundary_area[0][0], boundary_area[1][1]))
                    elif boundary_direction == 'right':
                        new_area = ((boundary_area[1][0], boundary_area[0][1]),
                                    (2*boundary_area[1][0] - boundary_area[0][0], boundary_area[1][1]))

                    area.append(new_area)
                    try:
                        captured_grims[area_index].append(new_area)
                    except ValueError:
                        captured_grims.append([new_area])
                    if area_index is not None:
                        update_single_area_cache(area_index, area)
                    return area

            # If the above fails, try folding each candidate in other directions (excluding the opposite of the candidate direction)
            opposite_direction = {
                'up': 'down',
                'down': 'up',
                'left': 'right',
                'right': 'left'
            }
            for boundary_area, boundary_direction in candidates:
                forbidden_direction = opposite_direction[boundary_direction]
                other_directions = [d for d in ['up', 'down', 'left', 'right'] if d != forbidden_direction and d != boundary_direction]
                random.shuffle(other_directions)
                for fold_direction in other_directions:
                    if is_valid_fold(boundary_area, fold_direction, exclude_region=area):
                        if fold_direction == 'up':
                            new_area = ((boundary_area[0][0], boundary_area[1][1]),
                                        (boundary_area[1][0], 2*boundary_area[1][1] - boundary_area[0][1]))
                        elif fold_direction == 'down':
                            new_area = ((boundary_area[0][0], 2*boundary_area[0][1] - boundary_area[1][1]),
                                        (boundary_area[1][0], boundary_area[0][1]))
                        elif fold_direction == 'left':
                            new_area = ((2*boundary_area[0][0] - boundary_area[1][0], boundary_area[0][1]),
                                        (boundary_area[0][0], boundary_area[1][1]))
                        elif fold_direction == 'right':
                            new_area = ((boundary_area[1][0], boundary_area[0][1]),
                                        (2*boundary_area[1][0] - boundary_area[0][0], boundary_area[1][1]))

                        area.append(new_area)
                        try:
                            captured_grims[area_index].append(new_area)
                        except ValueError:
                            captured_grims.append([new_area])
                        if area_index is not None:
                            update_single_area_cache(area_index, area)
                        return area

            # Current candidates cannot fold in any direction, try re-extracting
            attempt += 1

    # All sub-areas have been traversed and none can fold
    # Update cache (update even if no change, to ensure cache consistency)
    if area_index is not None:
        update_single_area_cache(area_index, area)
        # Mark this area as unable to expand further
        area_stats[area_index]['cannot_expand'] = True
    return area

def is_overlap(area1:tuple,area2:tuple):
    """
    Check whether two areas overlap.

    Args:
        area1 (list): The first area to check.
        area2 (list): The second area to check.

    Returns:
        bool: True if the areas overlap, False otherwise.
    """
    # Extract area coordinates
    (x1_min, y1_min), (x1_max, y1_max) = area1[0], area1[1]
    (x2_min, y2_min), (x2_max, y2_max) = area2[0], area2[1]

    # Check for overlap
    if (x1_min >= x2_max or x2_min >= x1_max or
        y1_min >= y2_max or y2_min >= y1_max):
        return False  # No overlap
    return True  # Has overlap

def is_within_bounds(area: tuple) -> bool:
    """
    Check whether the area is within the overall room boundary.

    Args:
        area (tuple): Area tuple ((x1, y1), (x2, y2))

    Returns:
        bool: True if within bounds, False if out of bounds
    """
    global global_room_side_length
    (x1, y1), (x2, y2) = area
    return (x1 >= 0 and y1 >= 0 and
            x2 <= global_room_side_length and
            y2 <= global_room_side_length)

def extract_whole_boundary(area_tuple):
    """
    Extract the four complete boundaries of an area (thickness 1, length equals the full edge length).

    Args:
        area_tuple (tuple): Area tuple ((x1, y1), (x2, y2))

    Returns:
        list: A list of four tuples, each being (boundary_area, direction),
              where boundary_area is a boundary strip area tuple ((bx1, by1), (bx2, by2)),
              and direction is the boundary direction ('up', 'down', 'left', 'right')
    """
    (x1, y1), (x2, y2) = area_tuple
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return []

    boundaries = []

    # Top edge: try thickness 2 first, then thickness 1
    top2 = ((x1, max(y1, y2 - 2)), (x2, y2))
    boundaries.append((top2, 'up'))
    top1 = ((x1, y2 - 1), (x2, y2))
    boundaries.append((top1, 'up'))

    # Bottom edge: try thickness 2 first, then thickness 1
    bottom2 = ((x1, y1), (x2, min(y2, y1 + 2)))
    boundaries.append((bottom2, 'down'))
    bottom1 = ((x1, y1), (x2, y1 + 1))
    boundaries.append((bottom1, 'down'))

    # Left edge: try thickness 2 first, then thickness 1
    left2 = ((x1, y1), (min(x2, x1 + 2), y2))
    boundaries.append((left2, 'left'))
    left1 = ((x1, y1), (x1 + 1, y2))
    boundaries.append((left1, 'left'))

    # Right edge: try thickness 2 first, then thickness 1
    right2 = ((max(x1, x2 - 2), y1), (x2, y2))
    boundaries.append((right2, 'right'))
    right1 = ((x2 - 1, y1), (x2, y2))
    boundaries.append((right1, 'right'))

    return boundaries

def extract_boundary_area(area_tuple):
    """
    Randomly extract a boundary strip from the area (fixed thickness 1, random length).

    Args:
        area_tuple (tuple): Area tuple ((x1, y1), (x2, y2))

    Returns:
        tuple: Boundary strip area tuple ((bx1, by1), (bx2, by2))
        str: Selected boundary direction ('up', 'down', 'left', 'right')

    """
    (x1, y1), (x2, y2) = area_tuple
    width = x2 - x1
    height = y2 - y1

    candidates = []

    # Top edge: thickness 2 (or as close to 2 as possible), then thickness 1 (random length and position)
    by1_2, by2_2 = max(y1, y2 - 2), y2
    sub_width = random.randint(1, max(1, width))
    bx1 = random.randint(x1, x2 - sub_width)
    candidates.append((((bx1, by1_2), (bx1 + sub_width, by2_2)), 'up'))

    by1_1, by2_1 = y2 - 1, y2
    sub_width = random.randint(1, max(1, width))
    bx1 = random.randint(x1, x2 - sub_width)
    candidates.append((((bx1, by1_1), (bx1 + sub_width, by2_1)), 'up'))

    # Bottom edge
    by1_2, by2_2 = y1, min(y2, y1 + 2)
    sub_width = random.randint(1, max(1, width))
    bx1 = random.randint(x1, x2 - sub_width)
    candidates.append((((bx1, by1_2), (bx1 + sub_width, by2_2)), 'down'))

    by1_1, by2_1 = y1, y1 + 1
    sub_width = random.randint(1, max(1, width))
    bx1 = random.randint(x1, x2 - sub_width)
    candidates.append((((bx1, by1_1), (bx1 + sub_width, by2_1)), 'down'))

    # Left edge
    bx1_2, bx2_2 = x1, min(x2, x1 + 2)
    sub_height = random.randint(1, max(1, height))
    by1 = random.randint(y1, y2 - sub_height)
    candidates.append((((bx1_2, by1), (bx2_2, by1 + sub_height)), 'left'))

    bx1_1, bx2_1 = x1, x1 + 1
    sub_height = random.randint(1, max(1, height))
    by1 = random.randint(y1, y2 - sub_height)
    candidates.append((((bx1_1, by1), (bx2_1, by1 + sub_height)), 'left'))

    # Right edge
    bx1_2, bx2_2 = max(x1, x2 - 2), x2
    sub_height = random.randint(1, max(1, height))
    by1 = random.randint(y1, y2 - sub_height)
    candidates.append((((bx1_2, by1), (bx2_2, by1 + sub_height)), 'right'))

    bx1_1, bx2_1 = x2 - 1, x2
    sub_height = random.randint(1, max(1, height))
    by1 = random.randint(y1, y2 - sub_height)
    candidates.append((((bx1_1, by1), (bx2_1, by1 + sub_height)), 'right'))

    # Ensure order: four thickness-2 (or near thickness-2) first, then four thickness-1 (current construction is already in order)
    return candidates

def calculate_area(area: list) -> float:
    """
    Calculate the total area. The area consists of multiple sub-rectangle diagonal tuples.

    Args:
        area (list): Area list, each element is ((x1,y1),(x2,y2))

    Returns:
        float: Total area
    """
    total = 0.0
    for subarea in area:
        (x1, y1), (x2, y2) = subarea
        total += (x2 - x1) * (y2 - y1)
    return total

def calculate_complexity(area: list) -> int:
    """
    Calculate the complexity of the area, represented by the number of sub-rectangles (fewer boundary corners).

    Args:
        area (list): Area list

    Returns:
        int: Number of sub-rectangles
    """
    return len(area)

def update_area_cache(areas: list):
    """
    Update cached data (area and complexity) for all areas.

    Args:
        areas (list): List of all areas
    """
    global area_stats
    for i, area in enumerate(areas):
        if i not in area_stats:
            area_stats[i] = {
                'selections': 0,
                'total_reward': 0.0,
                'area': 0.0,
                'complexity': 0,
                'last_reward': 0.0,
                'cannot_expand': False
            }
        # Update area and complexity cache
        area_stats[i]['area'] = calculate_area(area)
        area_stats[i]['complexity'] = calculate_complexity(area)

def update_single_area_cache(area_index: int, area: list):
    """
    Update cached data for a single area.

    Args:
        area_index (int): Area index
        area (list): Area data
    """
    global area_stats
    if area_index not in area_stats:
        area_stats[area_index] = {
            'selections': 0,
            'total_reward': 0.0,
            'area': 0.0,
            'complexity': 0,
            'last_reward': 0.0,
            'cannot_expand': False
        }
    # Update area and complexity cache
    area_stats[area_index]['area'] = calculate_area(area)
    area_stats[area_index]['complexity'] = calculate_complexity(area)

def calculate_rewards_from_cache():
    """
    Calculate rewards for all areas based on cached data.

    Returns:
        tuple: (rewards list, average area)
    """
    global area_stats
    if not area_stats:
        return [], 0.0

    # Get area and complexity for all areas
    areas_data = []
    for i, stats in area_stats.items():
        areas_data.append({
            'index': i,
            'area': stats['area'],
            'complexity': stats['complexity']
        })

    n = len(areas_data)
    # Calculate area range
    max_area = max(d['area'] for d in areas_data) if n > 0 else 1.0
    min_area = min(d['area'] for d in areas_data) if n > 0 else 1.0
    area_range = max_area - min_area
    if area_range == 0:
        area_range = 1.0

    # Calculate max and min for normalization
    max_complexity = max(d['complexity'] for d in areas_data) if n > 0 else 1.0
    min_complexity = min(d['complexity'] for d in areas_data) if n > 0 else 0.0
    complexity_range = max_complexity - min_complexity
    if complexity_range == 0:
        complexity_range = 1.0

    # Calculate rewards
    balance_weight = 0.5
    complexity_weight = 0.5
    rewards = []

    for d in areas_data:
        # Area balance reward
        norm_area = (d['area'] - min_area) / area_range
        balance_reward = 1.0 - norm_area

        # Complexity reward
        norm_complexity = (d['complexity'] - min_complexity) / complexity_range
        complexity_reward = 1.0 - norm_complexity

        # Combined reward
        reward = balance_weight * balance_reward + complexity_weight * complexity_reward
        rewards.append((d['index'], reward))
        # Update last_reward in cache
        area_stats[d['index']]['last_reward'] = reward

    # Sort by index and return
    rewards.sort(key=lambda x: x[0])
    return [r for _, r in rewards]

def choose_area_to_fold(areas:list):
    """
    Select an area to fold based on area balance and complexity, prioritizing areas
    with area close to the average and lower complexity.
    Uses the UCB algorithm to optimize selection, balancing exploration and exploitation,
    ensuring area differences between rooms are not too large.

    Args:
        areas (list): List of areas, each area is a list of sub-rectangle diagonal tuples

    Returns:
        int: Index of the selected area to fold, or -1 if no area can expand
    """
    global area_stats
    if not areas:
        return 0

    n = len(areas)

    # Ensure cache is up to date
    if not area_stats or len(area_stats) != n:
        update_area_cache(areas)

    # Calculate rewards based on cached data (average area not needed)
    rewards = calculate_rewards_from_cache()

    # Get indices of expandable areas
    expandable_indices = []
    for i in range(n):
        stats = area_stats.get(i, {'selections': 0, 'total_reward': 0.0, 'cannot_expand': False})
        if not stats.get('cannot_expand', False):
            expandable_indices.append(i)

    # If no expandable area, return -1
    if not expandable_indices:
        return -1

    # Calculate total selections (all areas, including non-expandable)
    total_selections = sum(stats.get('selections', 0) for stats in area_stats.values())

    # Calculate UCB score for each expandable area
    ucb_scores = []
    for i in expandable_indices:
        stats = area_stats.get(i, {'selections': 0, 'total_reward': 0.0})
        selections = stats.get('selections', 0)
        total_reward = stats.get('total_reward', 0.0)

        # Average reward
        avg_reward = total_reward / selections if selections > 0 else 0.0

        # UCB exploration term: sqrt(2 * ln(total_selections + 1) / (selections + 1e-6))
        # Use 1e-6 to avoid division by zero, encouraging exploration of less-selected areas
        exploration = (2 * math.log(total_selections + 1) / (selections + 1e-6)) ** 0.5
        ucb_score = avg_reward + exploration
        ucb_scores.append(ucb_score)

    # If all UCB scores are 0 (initial state), use raw rewards as selection criterion
    if all(s == 0 for s in ucb_scores):
        chosen_idx = max(expandable_indices, key=lambda i: rewards[i])
    else:
        # Find the expandable area index with the highest UCB score
        max_score_idx = max(range(len(ucb_scores)), key=lambda idx: ucb_scores[idx])
        chosen_idx = expandable_indices[max_score_idx]

    # Update statistics
    stats = area_stats.get(chosen_idx, {'selections': 0, 'total_reward': 0.0})
    stats['selections'] = stats.get('selections', 0) + 1
    # Use cached last_reward as the reward for this selection
    current_reward = area_stats[chosen_idx].get('last_reward', 0.0)
    stats['total_reward'] = stats.get('total_reward', 0.0) + current_reward
    area_stats[chosen_idx] = stats

    return chosen_idx

def is_valid_fold(area:tuple, direction:str, exclude_region=None):
    """
    Determine whether the fold is valid: check if the folded area overlaps with
    already occupied areas and does not exceed the room boundary.

    Args:
        area (tuple): The area to fold ((x1,y1),(x2,y2))
        direction (str): Fold direction ('up','down','left','right')
        exclude_area (tuple, optional): Area to exclude from overlap detection (e.g. parent rectangle)

    Returns:
        bool: True if the area is valid, False otherwise.
    """
    global global_room_side_length
    if direction not in ['up','down','left','right']:
        print("Invalid direction. Choose from 'up', 'down', 'left', 'right'.")
        return False
    if direction == 'up':
        # Check if the upper area overlaps with occupied areas
        new_area = ((area[0][0], area[1][1]), (area[1][0], 2*area[1][1] - area[0][1]))
        # Check if out of bounds
        if not is_within_bounds(new_area) or any(is_overlap(new_area, subarea) for region in captured_grims for subarea in region):
            return False
        # If the new area is a vertical strip with width 1, check if it can still fold left or right
        new_width = new_area[1][0] - new_area[0][0]
        if new_width == 1:
            # Simulate the new area from folding left
            left_new = ((2*new_area[0][0] - new_area[1][0], new_area[0][1]), (new_area[0][0], new_area[1][1]))
            left_ok = is_within_bounds(left_new) and not any(is_overlap(left_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(left_new, region) for region in exclude_region)
            # Simulate the new area from folding right
            right_new = ((new_area[1][0], new_area[0][1]), (2*new_area[1][0] - new_area[0][0], new_area[1][1]))
            right_ok = is_within_bounds(right_new) and not any(is_overlap(right_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(right_new, region) for region in exclude_region)
            if not left_ok and not right_ok:
                return False
        return True
    elif direction == 'down':
        # Check if the lower area overlaps with occupied areas
        new_area = ((area[0][0], 2*area[0][1] - area[1][1]), (area[1][0], area[0][1]))
        # Check if out of bounds
        if not is_within_bounds(new_area) or any(is_overlap(new_area, subarea) for region in captured_grims for subarea in region):
            return False
        # If the new area is a vertical strip with width 1, check if it can still fold left or right
        new_width = new_area[1][0] - new_area[0][0]
        if new_width == 1:
            left_new = ((2*new_area[0][0] - new_area[1][0], new_area[0][1]), (new_area[0][0], new_area[1][1]))
            left_ok = is_within_bounds(left_new) and not any(is_overlap(left_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(left_new, region) for region in exclude_region)
            right_new = ((new_area[1][0], new_area[0][1]), (2*new_area[1][0] - new_area[0][0], new_area[1][1]))
            right_ok = is_within_bounds(right_new) and not any(is_overlap(right_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(right_new, region) for region in exclude_region)
            if not left_ok and not right_ok:
                return False
        return True
    elif direction == 'left':
        # Check if the left area overlaps with occupied areas
        new_area = ((2*area[0][0] - area[1][0], area[0][1]), (area[0][0], area[1][1]))
        # Check if out of bounds
        if not is_within_bounds(new_area) or any(is_overlap(new_area, subarea) for region in captured_grims for subarea in region):
            return False
        # If the new area is a horizontal strip with height 1, check if it can still fold up or down
        new_height = new_area[1][1] - new_area[0][1]
        if new_height == 1:
            # Simulate the new area from folding up
            up_new = ((new_area[0][0], new_area[1][1]), (new_area[1][0], 2*new_area[1][1] - new_area[0][1]))
            up_ok = is_within_bounds(up_new) and not any(is_overlap(up_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(up_new, region) for region in exclude_region)
            # Simulate the new area from folding down
            down_new = ((new_area[0][0], 2*new_area[0][1] - new_area[1][1]), (new_area[1][0], new_area[0][1]))
            down_ok = is_within_bounds(down_new) and not any(is_overlap(down_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(down_new, region) for region in exclude_region)
            if not up_ok and not down_ok:
                return False
        return True
    elif direction == 'right':
        # Check if the right area overlaps with occupied areas
        new_area = ((area[1][0], area[0][1]), (2*area[1][0] - area[0][0], area[1][1]))
        # Check if out of bounds
        if not is_within_bounds(new_area) or any(is_overlap(new_area, subarea) for region in captured_grims for subarea in region):
            return False
        # If the new area is a horizontal strip with height 1, check if it can still fold up or down
        new_height = new_area[1][1] - new_area[0][1]
        if new_height == 1:
            up_new = ((new_area[0][0], new_area[1][1]), (new_area[1][0], 2*new_area[1][1] - new_area[0][1]))
            up_ok = is_within_bounds(up_new) and not any(is_overlap(up_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(up_new, region) for region in exclude_region)
            down_new = ((new_area[0][0], 2*new_area[0][1] - new_area[1][1]), (new_area[1][0], new_area[0][1]))
            down_ok = is_within_bounds(down_new) and not any(is_overlap(down_new, subarea) for region in captured_grims if region is not exclude_region for subarea in region) or any(is_overlap(down_new, region) for region in exclude_region)
            if not up_ok and not down_ok:
                return False
        return True


def merge_edges(all_edges):
    '''Merge adjacent edges in the edge list.'''
    # Separate into horizontal and vertical edges
    horizontal_edges = []
    vertical_edges = []
    for edge in all_edges:
        (x1, y1), (x2, y2) = edge
        if y1 == y2:  # Horizontal edge
            horizontal_edges.append(edge)
        elif x1 == x2:  # Vertical edge
            vertical_edges.append(edge)
    # Merge horizontal edges
    merged_horizontal = []
    horizontal_edges.sort(key=lambda e: (e[0][1], e[0][0]))  # Sort by y first, then by x
    for edge in horizontal_edges:
        if not merged_horizontal:
            merged_horizontal.append(edge)
        else:
            last_edge = merged_horizontal[-1]
            if last_edge[1] == edge[0]:  # On the same horizontal line and adjacent
                merged_horizontal[-1] = (last_edge[0], edge[1])  # Merge edges
            else:
                merged_horizontal.append(edge)
    # Merge vertical edges
    merged_vertical = []
    vertical_edges.sort(key=lambda e: (e[0][0], e[0][1]))  # Sort by x first, then by y
    for edge in vertical_edges:
        if not merged_vertical:
            merged_vertical.append(edge)
        else:
            last_edge = merged_vertical[-1]
            if last_edge[1] == edge[0]:  # On the same vertical line and adjacent
                merged_vertical[-1] = (last_edge[0], edge[1])  # Merge edges
            else:
                merged_vertical.append(edge)
    return merged_horizontal + merged_vertical

def merge_subarea():
    '''Merge all sub-areas of each room and generate boundary shapes.'''
    global captured_grims
    all_room = captured_grims[:]
    all_boundary = []
    # Only need to traverse all vertices of each room counterclockwise; the connected edges form the boundary
    for room in all_room:
        room_boundary = []
        for diagonals in room:
            add_to_room_boundary = []
            # Extract the four vertices of the diagonal
            (x1, y1), (x2, y2) = diagonals
            # Calculate four edges and check for overlap with room_boundary
            edges = {'left': ((x1, y1), (x1, y2)),  # Left edge
                     'top': ((x1, y2), (x2, y2)),  # Top edge
                     'right': ((x2, y1), (x2, y2)),  # Right edge
                     'bottom': ((x1, y1), (x2, y1))}  # Bottom edge

            for key, edge in edges.items():
                is_overlap = False
                if edge in room_boundary:
                    room_boundary.remove(edge)  # If the edge already exists, it is overlapping, remove it
                    # Also delete the edge from the dict
                    is_overlap = True
                    continue
                # Check for partial overlap
                elif key == 'left' or key == 'right':
                    for exist_edge in room_boundary:
                        if exist_edge[0][0] == exist_edge[1][0] and exist_edge[0][0] == edge[0][0]:  # Same x coordinate, possible partial overlap
                            if not (edge[1][1] <= exist_edge[0][1] or edge[0][1] >= exist_edge[1][1]):
                                is_overlap = True
                                room_boundary.remove(exist_edge)
                                # Keep the non-overlapping part
                                if edge[0][1] < exist_edge[0][1]:
                                    add_to_room_boundary.append((min(edge[0], exist_edge[0]), max(edge[0], exist_edge[0]))) if edge[0] != exist_edge[0] else None
                                    add_to_room_boundary.append((min(edge[1], exist_edge[1]), max(edge[1], exist_edge[1]))) if edge[1] != exist_edge[1] else None
                                else:
                                    add_to_room_boundary.append((min(exist_edge[0], edge[0]), max(exist_edge[0], edge[0]))) if exist_edge[0] != edge[0] else None
                                    add_to_room_boundary.append((min(exist_edge[1], edge[1]), max(exist_edge[1], edge[1]))) if exist_edge[1] != edge[1] else None
                elif key == 'top' or key == 'bottom':
                    for exist_edge in room_boundary:
                        if exist_edge[0][1] == exist_edge[1][1] and exist_edge[0][1] == edge[0][1]:  # Same y coordinate, possible partial overlap
                            if not (edge[1][0] <= exist_edge[0][0] or edge[0][0] >= exist_edge[1][0]):
                                is_overlap = True
                                room_boundary.remove(exist_edge)
                                # Keep the non-overlapping part
                                if edge[0][0] < exist_edge[0][0]:
                                    add_to_room_boundary.append((min(edge[0], exist_edge[0]), max(edge[0], exist_edge[0]))) if edge[0] != exist_edge[0] else None
                                    add_to_room_boundary.append((min(edge[1], exist_edge[1]), max(edge[1], exist_edge[1]))) if edge[1] != exist_edge[1] else None
                                else:
                                    add_to_room_boundary.append((min(exist_edge[0], edge[0]), max(exist_edge[0], edge[0]))) if exist_edge[0] != edge[0] else None
                                    add_to_room_boundary.append((min(exist_edge[1], edge[1]), max(exist_edge[1], edge[1]))) if exist_edge[1] != edge[1] else None
                if not is_overlap:
                    room_boundary.append(edge)  # Add non-overlapping edge
            room_boundary.extend(add_to_room_boundary)
            room_boundary = merge_edges(room_boundary)
        all_boundary.append(room_boundary)
    return all_boundary

def connectivity_record(all_boundary):
    '''Record room connectivity.'''
    # Record overlapping edges between different rooms; dict key is room pair, value is list of overlapping edges
    connectivity = {}
    for i in range(len(all_boundary)):
        for j in range(i+1, len(all_boundary)):
            # Check if the boundaries of two rooms overlap
            overlap_edges = []
            for edge1 in all_boundary[i]:
                for edge2 in all_boundary[j]:
                    if edge1 == edge2:
                        overlap_edges.append(edge1)
                        continue
                    # Check for partial overlap
                    # Extract edge endpoint coordinates
                    (x1, y1), (x2, y2) = edge1
                    (x3, y3), (x4, y4) = edge2
                    # Check if vertical edges (same x coordinate) on the same vertical line
                    if x1 == x2 and x3 == x4 and x1 == x3:
                        # y coordinates need sorting
                        y_min1, y_max1 = sorted([y1, y2])
                        y_min2, y_max2 = sorted([y3, y4])
                        # Check for overlap
                        if not (y_max1 <= y_min2 or y_max2 <= y_min1):
                            # Overlap exists, record the overlapping part
                            overlap_y_min = max(y_min1, y_min2)
                            overlap_y_max = min(y_max1, y_max2)
                            overlap_edges.append(((x1, overlap_y_min), (x1, overlap_y_max)))
                    # Check if horizontal edges (same y coordinate) on the same horizontal line
                    elif y1 == y2 and y3 == y4 and y1 == y3:
                        # x coordinates need sorting
                        x_min1, x_max1 = sorted([x1, x2])
                        x_min2, x_max2 = sorted([x3, x4])
                        # Check for overlap
                        if not (x_max1 <= x_min2 or x_max2 <= x_min1):
                            # Overlap exists, record the overlapping part
                            overlap_x_min = max(x_min1, x_min2)
                            overlap_x_max = min(x_max1, x_max2)
                            overlap_edges.append(((overlap_x_min, y1), (overlap_x_max, y1)))
            if overlap_edges:
                connectivity[(i, j)] = overlap_edges
    return connectivity

def check_graph_isomorphism(target_connectivity: list, generated_connectivity: dict, num_rooms: int) -> bool:
    '''Check whether two graphs are isomorphic.

    Args:
        target_connectivity: Target connectivity list, format like [(0,1), (1,2)], indicating connections between rooms 0-1 and 1-2
        generated_connectivity: Generated connectivity dict, format like {(0,1): [edge list], (1,2): [edge list]}
        num_rooms: Number of rooms

    Returns:
        bool: True if the two graphs are isomorphic, False otherwise
    '''
    # Convert target connectivity to adjacency matrix
    target_adj = [[0] * num_rooms for _ in range(num_rooms)]
    for i, j in target_connectivity:
        if 0 <= i < num_rooms and 0 <= j < num_rooms:
            target_adj[i][j] = 1
            target_adj[j][i] = 1

    # Convert generated connectivity to adjacency matrix
    generated_adj = [[0] * num_rooms for _ in range(num_rooms)]
    for (i, j) in generated_connectivity.keys():
        if 0 <= i < num_rooms and 0 <= j < num_rooms:
            generated_adj[i][j] = 1
            generated_adj[j][i] = 1

    # If the two graphs have different edge counts, they are definitely not isomorphic
    target_edges = sum(sum(row) for row in target_adj) // 2
    generated_edges = sum(sum(row) for row in generated_adj) // 2
    if target_edges != generated_edges:
        return False

    # Check if degree sequences are the same
    target_degrees = sorted(sum(row) for row in target_adj)
    generated_degrees = sorted(sum(row) for row in generated_adj)
    if target_degrees != generated_degrees:
        return False

    # For small graphs (n <= 10), use brute-force to check isomorphism
    if num_rooms <= 10:
        # Generate all possible node mappings
        rooms = list(range(num_rooms))
        for perm in itertools.permutations(rooms):
            # Check if this mapping preserves adjacency
            is_isomorphic = True
            for i in range(num_rooms):
                for j in range(num_rooms):
                    if target_adj[i][j] != generated_adj[perm[i]][perm[j]]:
                        is_isomorphic = False
                        break
                if not is_isomorphic:
                    break
            if is_isomorphic:
                return True
        return False
    else:
        return False  # Large graphs not handled yet
