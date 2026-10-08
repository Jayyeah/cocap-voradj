import numpy as np

from utils.logger_setter import logger as logger


def find_robot_by_id(robot_id, robot_list):
    """
    Find a robot object in a list based on its ID.

    This function iterates through the given list of robots to find an object matching the specified ID.
    If a matching robot object is found, it is returned; otherwise, a runtime error is raised.

    Args:
        robot_id (int | signedinteger): The ID of the robot to find.
        robot_list (list): A list of robot objects, each expected to have an `id` attribute.

    Returns:
        Pursuer | Evader: The robot object matching the specified ID.

    Raises:
        RuntimeError: If no matching robot object is found in the list.

    """
    try:
        if robot_id is None:
            raise ValueError("robot_id cannot be None!")
        if not isinstance(robot_id, int | np.signedinteger):
            raise ValueError("robot_id must be an integer!")

        for robot in robot_list:
            if robot_id == robot.id:
                # logger.info(f"Finding robot with ID: {robot_id}!!!!!!!!=================")
                # print(f"Finding robot with ID: {robot_id}!!!!!!!!=================")
                return robot
        raise AttributeError(f"Wrong robot_id: {robot_id} input!")
    except Exception as e:
        logger.error(f"An error occurred in find_robot_by_id: {e}")
        raise RuntimeError("Robot not found in robot_list!")


def calculate_angle_from_components(v1: float, v2: float) -> list:
    """
    Calculate the angle of a velocity vector in a 2D coordinate system.

    This function takes the x and y components of a velocity vector, computes the angle (in radians)
    between the positive x-axis and the vector, and normalizes the angle to the range [0, 2π].

    Args:
        v1 (float): The x-component of the velocity.
        v2 (float): The y-component of the velocity.

    Returns:
        list of float: The angle (in radians) between the positive x-axis and the velocity vector,
                       normalized to the range [0, 2π].

    Examples:
        >>> v_1 = 1.0
        >>> v_2 = 1.0
        >>> calculate_angle_from_components(v_1, v_2)
        0.7853981633974483  # Approximately π/4 radians or 45 degrees
    """
    # Calculate the angle corresponding to the velocity components
    theta = np.arctan2(v2, v1)

    # Normalize the angle to the range [0, 2π]
    if theta < 0:
        theta += 2 * np.pi
    return [theta]


def to_list(input_data):
    """
    Convert any input data into a list.

    Args:
        input_data: Can be a scalar, iterable, or NumPy array.

    Returns:
        list: The converted list.
    """
    if isinstance(input_data, np.ndarray):
        # If it's a NumPy array, convert using tolist()
        return input_data.tolist()
    elif isinstance(input_data, (list, tuple, set)):
        # If it's already a list, tuple, or set, convert to a list
        return list(input_data)
    elif isinstance(input_data, (np.integer, np.floating, int, float, str)):
        # If it's a single scalar value (including NumPy scalars and Python basic types), wrap in a list
        return [input_data]
    else:
        # For other iterable objects (e.g., dictionaries), convert their keys to a list
        try:
            return list(input_data)
        except TypeError:
            # If it cannot be iterated, treat it as a single element in a list
            return [input_data]
