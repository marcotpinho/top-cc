import random
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist

import plot
from src.entities.Archive import Archive
from src.entities.Evaluator import Evaluator
from src.entities.MOVNS import MOVNS
from src.entities.Map import Map
from src.entities.Neighborhood import Neighborhood
from src.entities.Solution import Solution
from src.operators.local_search import local_search

DEFAULT_OUTPUT_DIR = "imgs/"
DEFAULT_PATHS_SUBDIR = "paths/"


def run_optimization(
    rpositions: np.ndarray, # shape (n, 2)
    rvalues: np.ndarray, # shape (n,)
    budget: list[int],
    map_name: str,
    output_dir: str,
    begin: int = -1,
    end: int = -2,
    total_time: int = 600,
    num_agents: int = 1,
    speeds: list = [1],
    seed: int = 42,
    max_iterations: int = 100,
    algorithm: str = "unique_vis",
    save_results: bool = True,
    plot_results: bool = True,
) -> list:
    """
    Run the complete multi-objective optimization pipeline.
    
    Args:
        rpositions: Coordinates of reward nodes
        rvalues: Reward values for each node
        budget: Budget constraints for each agent
        map_name: Name of the map being used
        output_dir: Directory to save results
        begin: Index of the starting node
        end: Index of the ending node
        total_time: Maximum execution time in seconds
        num_agents: Number of agents in the system
        speeds: Speed values for each agent
        seed: Random seed for reproducibility
        max_iterations: Maximum number of iterations
        algorithm: Algorithm variant to use
        save_results: Whether to save results to files
        plot_results: Whether to generate plots
        
    Returns:
        List of Pareto optimal solution paths
    """
    Solution.set_parameters(begin, end, num_agents, budget, speeds)

    # Calculate distance matrix between all reward positions
    distance_matrix = cdist(rpositions, rpositions, metric="euclidean")

    archive, front, log = run_movns(
        rvalues, rpositions, distance_matrix, total_time, seed, max_iterations, algorithm
    )
    
    archive.sort(key=lambda solution: solution.score[0])

    paths = [solution.get_solution_paths() for solution in front]
    scores = np.array([solution.score for solution in front])

    if scores.size > 0:
        print(f"Best reward score: {max(scores[:, 0]):.2f}")
    else:
        print("No solutions found in Pareto front")

    if save_results and paths:
        save_results_to_files(paths, scores, log, output_dir, map_name, max(speeds))

    if plot_results and paths:
        plot_solution_paths(paths, scores, rpositions, rvalues, map_name)

    return paths


def run_movns(
    rvalues: np.ndarray, # shape (n,)
    rpositions: np.ndarray, # shape (n, 2)
    distmx: np.ndarray, # shape (n, n)
    total_time: int,
    seed: int,
    max_iterations: int = 100,
    algorithm: str = "unique_vis",
) -> tuple[list, list, list]:
    """
    Run the Multi-Objective Variable Neighborhood Search algorithm.
    
    Args:
        rvalues: Reward values for each node
        rpositions: Positions of reward nodes
        distmx: Distance matrix between nodes
        total_time: Maximum execution time in seconds
        seed: Random seed for reproducibility
        max_iterations: Maximum number of iterations
        algorithm: Algorithm variant to use
        
    Returns:
        Tuple of (archive, front, log)
    """
    np.random.seed(seed)
    random.seed(seed)

    mapp = Map(rvalues=rvalues, rpositions=rpositions, distmx=distmx)
    neighborhood  = Neighborhood(algorithm)
    archive = Archive()
    evaluator = Evaluator(map=mapp)
    movns = MOVNS(max_iterations=max_iterations,
                  max_time=total_time,
                  archive=archive,
                  neighborhood=neighborhood,
                  map=mapp,
                  evaluator=evaluator)

    log, elapsed_time = movns.run()

    print(f"Finished running in {elapsed_time:.2f} seconds.")

    return archive.front + archive.dominated, archive.front, log


def save_results_to_files(
    paths: list, 
    scores: np.ndarray, 
    log: list,
    output_dir: str, 
    map_name: str, 
    max_speed: float
) -> None:
    """
    Save optimization results to pickle files.
    
    Args:
        paths: List of solution paths
        scores: Array of solution scores
        log: Optimization log
        output_dir: Base output directory
        map_name: Name of the map used
        max_speed: Maximum speed value
    """
    results_dir = Path(output_dir) / map_name / str(max_speed)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for path, score in zip(paths, scores):
        with open(results_dir / "scores.pkl", "ab") as f:
            pickle.dump(score, f)
        with open(results_dir / "paths.pkl", "ab") as f:
            pickle.dump(path, f)
    
    with open(results_dir / "log.pkl", "ab") as f:
        pickle.dump(log, f)


def plot_solution_paths(
    paths: list,
    scores: np.ndarray,
    rpositions: np.ndarray, # shape (n, 2)
    rvalues: np.ndarray, # shape (n,)
    map_name: str,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> None:
    """
    Plot and save visualization of solution paths.
    
    Args:
        paths: List of solution paths
        scores: Array of solution scores
        rpositions: Positions of reward nodes
        rvalues: Reward values
        map_name: Name of the map
        output_dir: Output directory for plots
    """
    print("Plotting paths...")
    
    plots_dir = Path(output_dir) / DEFAULT_PATHS_SUBDIR / map_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    existing_files = len([f for f in plots_dir.iterdir() if f.is_file()])
    
    for i, (path, score) in enumerate(zip(paths, scores)):
        filename = str(existing_files + i + 1)
        plot.plot_paths_with_rewards(
            rpositions,
            rvalues,
            path,
            score,
            directory=str(plots_dir),
            fname=filename,
        )
