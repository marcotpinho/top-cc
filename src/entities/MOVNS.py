from tqdm import tqdm
import time

from src.movns import select_solution_from_front

class MOVNS:
    def __init__(self, value: float):
        self.value = value
        self.archive = None
        self.map = None
        self.max_iterations = None
        self.max_time = None
        self.neighborhood = None
        self.log = []
    

    def run(self):
        """Run the main optimization loop."""
        start_time = time.perf_counter()

        progress_bar = tqdm(total=self.max_iterations, desc="Progress", unit="it", dynamic_ncols=True)
        iteration = 0

        while iteration < self.max_iterations and time.perf_counter() - start_time < self.max_time:
            solution = select_solution_from_front(self.archive.front, self.archive.dominated, iteration)

            for neighborhood_id in range(self.neighborhood.num_neighborhoods):
                if time.perf_counter() - start_time > self.max_time:
                    break

                perturbed_solution = perturb_solution(
                    solution, self.neighborhood, self.map.rvalues, self.map.rpositions, self.map.distmx
                )
                neighbors = local_search(
                    perturbed_solution, self.neighborhood, neighborhood_id, self.map.rvalues, self.map.rpositions, self.map.distmx
                )
                new_solutions = [perturbed_solution] + neighbors
                evaluate(new_solutions, self.map.rvalues, self.map.rpositions, self.map.distmx)

                self.archive, self.front, self.dominated = update_archive(
                    self.archive, [perturbed_solution] + neighbors, ARCHIVE_MAX_SIZE
                )

                save_statistics(self.log, self.front)

            iteration += 1
            progress_bar.update(1)

        progress_bar.close()

        elapsed_time = time.perf_counter() - start_time

        return archive, front, log

    def __repr__(self):
        return f"MOVNS(value={self.value})"