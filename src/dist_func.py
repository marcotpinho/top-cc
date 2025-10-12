import numpy as np
import torch
import torch.nn as nn

from .entities.DistanceModel import DistanceModel

MODEL = None
MODEL_DEVICE = None
EPSILON = 1e-8

def load_model(model_path: str = "checkpoints/model.pth"):
    global MODEL, MODEL_DEVICE

    if MODEL is None:
        MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL = DistanceModel(device=MODEL_DEVICE).to(MODEL_DEVICE)
        MODEL.load_state_dict(torch.load(model_path, map_location=MODEL_DEVICE))
        MODEL.eval()
        print(f"Model loaded on {MODEL_DEVICE}")
    
    return MODEL

def predict_max_distance(coords, timestamps) -> float:
    model = load_model()
    batch_data = transform_input(coords, timestamps)
    with torch.no_grad():
        prediction = model(batch_data)
        
        map_diag = batch_data["map_diag"]
        max_distance = prediction * map_diag
        
    return max_distance.item()


def transform_input(coords, timestamps) -> dict:
    coords_normalized, map_diag = normalize_coordinates(coords)
    coords_tensors = [torch.tensor(c, dtype=torch.float32) for c in coords_normalized]

    times_normalized, _, _ = normalize_timestamps(timestamps)
    times_tensors = [torch.tensor(ts, dtype=torch.float32).unsqueeze(-1) for ts in times_normalized]

    points_normalized = [torch.hstack((c, ts)) for c, ts in zip(coords_tensors, times_tensors)]

    lengths = torch.tensor([len(path) for path in points_normalized])
    instance_ids = torch.tensor([0] * len(points_normalized))
    
    padded = nn.utils.rnn.pad_sequence(points_normalized, batch_first=True)

    return {
        "points": padded,
        "lengths": lengths,
        "instance_ids": instance_ids,
        "map_diag": torch.tensor(map_diag, dtype=torch.float32)
    }

def normalize_coordinates(coordinates: list) -> tuple[list, float]:
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


def normalize_timestamps(timestamps: list) -> tuple[list, float, float]:
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


def test_prediction():
    """Test the prediction function with sample data."""
    # Sample data
    coords = [
        np.array([[0, 0], [1, 1], [2, 2]]),
        np.array([[3, 3], [4, 4], [5, 5]]),
    ]
    timestamps = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    ]
    
    try:
        load_model()
        distance = predict_max_distance(coords, timestamps)
        print(f"Predicted max distance: {distance:.4f}")
        return distance
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    test_prediction()