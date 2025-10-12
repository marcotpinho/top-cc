import numpy as np
import torch
import torch.nn as nn
from typing import List, Union
from .entities.DistanceModel import DistanceModel

MODEL = None
MODEL_DEVICE = None
EPSILON = 1e-8

def load_model(model_path: str = "checkpoints/model.pth"):
    global MODEL, MODEL_DEVICE

    if MODEL is None:
        MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load complete model instead of state dict for better deployment
        MODEL = DistanceModel(device=MODEL_DEVICE).to(MODEL_DEVICE)
        MODEL.load_state_dict(torch.load(model_path, map_location=MODEL_DEVICE))
        print(f"Model loaded on {MODEL_DEVICE}")
    
    return MODEL

def predict_max_distance(coords, timestamps) -> float:
    """Single instance prediction (backward compatibility)."""
    results = predict_max_distance_batch([coords], [timestamps])
    return results[0]

def predict_max_distance_batch(
    coords_batch: List[List[np.ndarray]], 
    timestamps_batch: List[List[np.ndarray]]
) -> List[float]:
    """
    Predict max distances for a batch of optimization instances.
    
    Args:
        coords_batch: List of instances, each containing list of agent coordinates
        timestamps_batch: List of instances, each containing list of agent timestamps
    
    Returns:
        List of predicted max distances (one per instance)
    
    Example:
        coords_batch = [
            [np.array([[0,0], [1,1]]), np.array([[2,2], [3,3]])],  # Instance 1: 2 agents
            [np.array([[4,4], [5,5]]), np.array([[6,6], [7,7]])]   # Instance 2: 2 agents  
        ]
    """
    model = load_model()
    
    if len(coords_batch) == 0:
        return []
    
    # Collect all paths from all instances
    all_coords = []
    all_timestamps = []
    instance_ids = []
    instance_map_diags = []
    
    for instance_id, (coords, timestamps) in enumerate(zip(coords_batch, timestamps_batch)):
        if len(coords) == 0:
            instance_map_diags.append(1.0)
            continue
            
        # Calculate map diagonal for this instance
        coords_flat = np.array([c for c_list in coords for c in c_list])
        if len(coords_flat) > 0:
            max_vals = coords_flat.max(axis=0)
            min_vals = coords_flat.min(axis=0) 
            map_diag = np.linalg.norm(max_vals - min_vals) + EPSILON
        else:
            map_diag = 1.0
        instance_map_diags.append(map_diag)
        
        # Add this instance's data
        all_coords.extend(coords)
        all_timestamps.extend(timestamps)
        instance_ids.extend([instance_id] * len(coords))
    
    if len(all_coords) == 0:
        return [0.0] * len(coords_batch)
    
    # Transform inputs using the same logic as training
    batch_data = transform_input_batch(all_coords, all_timestamps, instance_ids)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(batch_data)  # Shape: [num_instances]
    
    # Denormalize predictions using each instance's map diagonal
    results = []
    for i, pred in enumerate(predictions):
        if i < len(instance_map_diags):
            denormalized = pred.item() * instance_map_diags[i]
            results.append(denormalized)
        else:
            results.append(0.0)
    
    return results

def transform_input_batch(coords, timestamps, instance_ids) -> dict:
    """Transform batch input data for the model."""
    coords_normalized, global_map_diag = normalize_coordinates_batch(coords)
    coords_tensors = [torch.tensor(c, dtype=torch.float32) for c in coords_normalized]

    times_normalized, _, _ = normalize_timestamps_batch(timestamps)
    times_tensors = [torch.tensor(ts, dtype=torch.float32).unsqueeze(-1) for ts in times_normalized]

    points_normalized = [torch.hstack((c, ts)) for c, ts in zip(coords_tensors, times_tensors)]

    lengths = torch.tensor([len(path) for path in points_normalized])
    instance_ids_tensor = torch.tensor(instance_ids)
    
    padded = nn.utils.rnn.pad_sequence(points_normalized, batch_first=True)

    return {
        "points": padded.to(MODEL_DEVICE),
        "lengths": lengths.to(MODEL_DEVICE),
        "instance_ids": instance_ids_tensor.to(MODEL_DEVICE),
        "map_diag": torch.tensor(global_map_diag, dtype=torch.float32).to(MODEL_DEVICE)
    }

def normalize_coordinates_batch(coordinates: list) -> tuple[list, float]:
    """Normalize coordinates across entire batch."""
    if not coordinates or len(coordinates) == 0:
        return [], 1.0
    
    coords_flat = np.array([c for coords in coordinates for c in coords])
    
    if len(coords_flat) == 0:
        return [], 1.0
        
    max_vals = coords_flat.max(axis=0)
    min_vals = coords_flat.min(axis=0)
    map_diag = np.linalg.norm(max_vals - min_vals) + EPSILON
    
    center = (max_vals + min_vals) / 2
    
    coords_normalized = []
    for coords in coordinates:
        normalized = (coords - center) / map_diag
        coords_normalized.append(normalized)
    
    return coords_normalized, map_diag

def normalize_timestamps_batch(timestamps: list) -> tuple[list, float, float]:
    """Normalize timestamps across entire batch."""
    if not timestamps or len(timestamps) == 0:
        return [], 0.0, 1.0
    
    times_flat = np.array([ts for times in timestamps for ts in times])
    t_max = times_flat.max()
    t_min = times_flat.min()
    time_range = t_max - t_min + EPSILON
    
    times_normalized = []
    for ts in timestamps:
        normalized = (ts - t_min) / time_range
        times_normalized.append(normalized)
    
    return times_normalized, t_max, t_min

# Legacy functions for backward compatibility
def transform_input(coords, timestamps) -> dict:
    """Single instance transform (backward compatibility)."""
    return transform_input_batch(coords, timestamps, [0] * len(coords))

def normalize_coordinates(coordinates: list) -> tuple[list, float]:
    """Single instance coordinate normalization (backward compatibility)."""
    return normalize_coordinates_batch(coordinates)

def normalize_timestamps(timestamps: list) -> tuple[list, float, float]:
    """Single instance timestamp normalization (backward compatibility)."""
    return normalize_timestamps_batch(timestamps)

def test_prediction():
    """Test both single and batch prediction."""
    # Sample data for single prediction
    coords = [
        np.array([[0, 0], [1, 1], [2, 2]]),
        np.array([[3, 3], [4, 4], [5, 5]]),
    ]
    timestamps = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    ]
    
    try:
        # Test single prediction
        distance = predict_max_distance(coords, timestamps)
        print(f"Single prediction: {distance:.4f}")
        
        # Test batch prediction
        coords_batch = [coords, coords]  # Same instance twice
        timestamps_batch = [timestamps, timestamps]
        
        batch_distances = predict_max_distance_batch(coords_batch, timestamps_batch)
        print(f"Batch predictions: {[f'{d:.4f}' for d in batch_distances]}")
        
        return distance, batch_distances
    except Exception as e:
        print(f"Test failed: {e}")
        return None, None

if __name__ == "__main__":
    test_prediction()