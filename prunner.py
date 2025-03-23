def get_correct_candidates(instance, cell_size=10):
    """
    Given an instance (a list of attribute values from the ARFF data),
    compute the safe moves and select those that yield the minimum Manhattan distance
    to the food.

    Instance indices (0-indexed):
      0: head_x, 1: head_y,
      2: food_x, 3: food_y,
      4: food_left, 5: food_up, 6: food_right, 7: food_down,  (we don't use these in this function)
      8: left_occ, 9: up_occ, 10: right_occ, 11: down_occ,
      12: direction (class label)

    Returns a set of candidate directions that are safe and bring the snake closest to the food.
    """
    # Read positions as floats (or ints)
    head_x = float(instance[0])
    head_y = float(instance[1])
    food_x = float(instance[2])
    food_y = float(instance[3])

    # Occupancy flags (0 means free, 1 means blocked)
    left_occ = int(instance[8])
    up_occ = int(instance[9])
    right_occ = int(instance[10])
    down_occ = int(instance[11])

    # Define possible moves with the corresponding change in coordinates.
    moves = {
        "UP": (0, -cell_size),
        "DOWN": (0, cell_size),
        "LEFT": (-cell_size, 0),
        "RIGHT": (cell_size, 0)
    }

    safe_moves = {}

    # For each direction, if the move is safe, compute the Manhattan distance after moving.
    for d, (dx, dy) in moves.items():
        if d == "LEFT" and left_occ == 1:
            continue
        if d == "RIGHT" and right_occ == 1:
            continue
        if d == "UP" and up_occ == 1:
            continue
        if d == "DOWN" and down_occ == 1:
            continue

        new_x = head_x + dx
        new_y = head_y + dy
        new_distance = abs(new_x - food_x) + abs(new_y - food_y)
        safe_moves[d] = new_distance

    if not safe_moves:
        # No safe moves found: return an empty set so the instance isn't pruned automatically.
        return set()

    # Find the minimal Manhattan distance among safe moves.
    min_distance = min(safe_moves.values())
    # Consider as correct candidates only the safe moves with this minimal distance.
    candidates = {d for d, dist in safe_moves.items() if dist == min_distance}
    return candidates


def prune_arff_instances(input_filename, output_filename, cell_size=10):
    """
    Reads an ARFF file and writes a pruned version to output_filename.
    For each instance, the function calculates the safe candidate directions
    (i.e. moves that are not blocked and that yield the minimum Manhattan distance to the food).
    The instance is kept if its class label (the chosen direction) is among these candidates,
    or if no safe candidate can be computed.
    """
    header_lines = []
    data_lines = []
    in_data_section = False

    with open(input_filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("@data"):
                in_data_section = True
                header_lines.append(stripped)
                continue
            if not in_data_section:
                header_lines.append(stripped)
            else:
                data_lines.append(stripped)

    pruned_data = []
    num_total = 0
    num_kept = 0

    for dline in data_lines:
        instance = [x.strip() for x in dline.split(",")]
        # Ensure instance has the expected number of attributes (13 in this case)
        if len(instance) < 20:
            continue
        num_total += 1

        # Compute correct candidate directions
        candidates = get_correct_candidates(instance, cell_size=cell_size)
        chosen_direction = instance[19]

        # If there are safe candidate moves, keep the instance only if the chosen move is among them.
        # If no candidates could be computed (e.g. all adjacent moves blocked), keep the instance.
        if candidates:
            if chosen_direction in candidates:
                pruned_data.append(dline)
                num_kept += 1
        else:
            pruned_data.append(dline)
            num_kept += 1

    # Write the pruned ARFF file.
    with open(output_filename, "w") as f:
        for h in header_lines:
            f.write(h + "\n")
        for d in pruned_data:
            f.write(d + "\n")

    print(f"Pruned {num_total} instances to {num_kept} instances.")


# Example usage:
prune_arff_instances("Arffs/extended_binary.arff", "Arffs/clean_extended.arff")