import numpy as np
from numba import njit
from typing import List

from db_utils import save_to_db
from utils import calculate_rssi
from .dist_func_batch import predict_max_distance_batch
from .entities import Solution


def evaluate_batch(
    solutions: List[Solution],
    rvalues: np.ndarray, # shape (n,)
    rpositions: np.ndarray, # shape (n, 2)
    distmx: np.ndarray, # shape (n, n)
) -> List[tuple[float, float, float]]:
    """
    Evaluate multiple solutions in batch for better performance.
    
    Args:
        solutions: List of Solution objects to evaluate
        rvalues: Reward values for each node
        rpositions: Positions of reward nodes  
        distmx: Distance matrix between nodes
        
    Returns:
        List of (max_reward, max_rssi, neg_min_len) tuples
    """
    if len(solutions) == 0:
        return []
    
    # Collect all solution data for batch processing
    all_paths = []
    all_coordinates = []
    all_timestamps = []
    solution_results = []
    
    speeds = np.array(Solution.speeds)
    
    for solution in solutions:
        paths = solution.get_solution_paths()
        paths_flat = np.concatenate(paths)
        
        # Calculate reward (fast, can do individually)
        max_reward = maximize_reward(paths_flat, rvalues)
        
        # Calculate path length (fast, can do individually)  
        min_len = get_paths_max_length(paths, distmx)
        
        # Prepare data for batch distance prediction
        interesting_times, timestamps = get_time_to_rewards(paths, speeds, distmx)
        coordinates = [rpositions[path] for path in paths]
        
        all_coordinates.append(coordinates)
        all_timestamps.append(timestamps)
        
        # Store partial results
        solution_results.append({
            'max_reward': max_reward,
            'min_len': min_len,
            'paths': paths
        })
    
    # Batch predict max distances
    max_distances = predict_max_distance_batch(all_coordinates, all_timestamps)
    
    # Complete the evaluation for each solution
    final_results = []
    for i, (max_distance, result) in enumerate(zip(max_distances, solution_results)):
        max_rssi = calculate_rssi(max_distance, noise=False)
        
        # Optionally save to database
        # if np.random.random() < 0.1:
        #     save_to_db(result['paths'], speeds, rpositions, max_distance)
        
        final_results.append((result['max_reward'], max_rssi, -result['min_len']))
    
    return final_results


def evaluate(
    solution: Solution,
    rvalues: np.ndarray, # shape (n,)
    rpositions: np.ndarray, # shape (n, 2)
    distmx: np.ndarray, # shape (n, n)
) -> tuple[float, float, float]:
    """
    Single solution evaluation (backward compatibility).
    For better performance, use evaluate_batch() when possible.
    """
    results = evaluate_batch([solution], rvalues, rpositions, distmx)
    return results[0] if results else (0.0, 0.0, 0.0)


@njit(cache=True, fastmath=True)
def get_paths_max_length_numba(paths_array: np.ndarray, distmx: np.ndarray) -> float:
    """Numba-optimized version for array input."""
    max_distance = 0.0
    
    for i in range(len(paths_array)):
        path = paths_array[i]
        distances = 0.0
        for j in range(len(path) - 1):
            distances += distmx[path[j], path[j + 1]]
        if distances > max_distance:
            max_distance = distances
            
    return max_distance


def get_paths_max_length(paths: list[np.ndarray], distmx: np.ndarray) -> float:
    """Calculate maximum path length across all paths."""
    max_distance = 0.0
    
    for path in paths:
        distances = 0.0
        for j in range(len(path) - 1):
            distances += distmx[path[j], path[j + 1]]
        max_distance = max(max_distance, distances)
            
    return max_distance


def interpolate_positions(
    paths: list[np.ndarray],
    speeds: np.ndarray, # shape (k,)
    interesting_times: np.ndarray,
    rpositions: np.ndarray, # shape (n, 2)
    distmx: np.ndarray, # shape (n, n)
) -> np.ndarray:
    """Interpolate agent positions at interesting time points."""
    num_paths = len(paths)
    num_times = len(interesting_times)
    interpolated_positions = np.zeros((num_paths, num_times, 2))
    
    for i in range(num_paths):
        path = paths[i]
        speed = speeds[i]

        if len(path) <= 1:
            continue
            
        time_to_rewards = np.zeros(len(path) - 1)
        cumulative_dist = 0.0
        for j in range(len(path) - 1):
            cumulative_dist += distmx[path[j], path[j + 1]]
            time_to_rewards[j] = cumulative_dist / speed
        
        x_positions = np.zeros(len(path) - 1)
        y_positions = np.zeros(len(path) - 1)
        for j in range(len(path) - 1):
            x_positions[j] = rpositions[path[j + 1], 0]
            y_positions[j] = rpositions[path[j + 1], 1]
        
        # Interpolate for each interesting time
        for t in range(num_times):
            if len(time_to_rewards) > 0:
                interpolated_positions[i, t, 0] = np.interp(interesting_times[t], time_to_rewards, x_positions)
                interpolated_positions[i, t, 1] = np.interp(interesting_times[t], time_to_rewards, y_positions)
    
    return interpolated_positions


@njit(cache=True, fastmath=True)
def get_time_to_rewards(
    paths: list[np.ndarray],
    speeds: np.ndarray, # shape (k,)
    distmx: np.ndarray # shape (n, n)
) -> tuple[np.ndarray, list[list[float]]]:
    """Calculate timestamps for reward collection along paths."""
    all_times = []
    timestamps = []
    
    for i in range(len(paths)):
        path = paths[i]
        speed = speeds[i]
        path_times = [0.0]  # Start with time 0.0
        
        cumulative_dist = 0.0
        for j in range(len(path) - 1):
            cumulative_dist += distmx[path[j], path[j + 1]]
            ts = cumulative_dist / speed
            path_times.append(ts)
            all_times.append(ts)
        
        timestamps.append(path_times)

    if len(all_times) == 0:
        return np.array([0.0]), timestamps
    
    times_array = np.array(all_times)
    return np.unique(times_array), timestamps


@njit(cache=True, fastmath=True)
def calculate_max_distance(interpolated_positions: np.ndarray) -> float:
    """Calculate maximum distance between any two agents at any time."""
    if len(interpolated_positions) <= 1:
        return 0.0
    
    k, n, _ = interpolated_positions.shape
    max_distance = 0.0
    
    for i in range(k):
        for j in range(i + 1, k):
            for t in range(n):
                pos_i = interpolated_positions[i, t, :]
                pos_j = interpolated_positions[j, t, :]
                
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                max_distance = max(max_distance, distance) 
    
    return max_distance


@njit(cache=True, fastmath=True)
def maximize_reward(paths_flat: np.ndarray, rvalues: np.ndarray) -> float:
    """Calculate total reward from unique visited nodes."""
    unique_elements = np.unique(paths_flat)
    reward = 0.0
    for element in unique_elements:
        reward += rvalues[element]
    return reward


