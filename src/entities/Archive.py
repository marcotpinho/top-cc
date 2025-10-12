from src.entities import Solution
from src.operators.local_search import local_search

ARCHIVE_MAX_SIZE = 40

class Archive:
    def __init__(self, evaluator = None, neighborhood = None, map = None, max_size: int = ARCHIVE_MAX_SIZE):
        self.front = []
        self.dominated = []
        self.max_size = max_size

        self.evaluator = evaluator
        self.neighborhood = neighborhood
        self.map = map

    def populate(self, initial_solution: Solution):
        for neighborhood_id in range(self.neighborhood.num_neighborhoods):
            neighbors = local_search(
                initial_solution,
                self.neighborhood,
                neighborhood_id,
                self.map.rvalues,
                self.map.rpositions,
                self.map.distmx
            )
            self.evaluator.evaluate(neighbors, self.map.rvalues, self.map.rpositions, self.map.distmx)
            self.update_archive(neighbors)

    def update_archive(self, neighbors: list[Solution]) -> None:
        """Update archive with non-dominated solutions."""
        all_solutions = self.front + self.dominated + neighbors

        self.front, all_dominated = self._get_non_dominated_solutions(all_solutions)

        if len(self.front) > self.max_size:
            self.front = self._select_by_crowding_distance(self.front, self.max_size)

        self.dominated = []
        if len(self.front) < self.max_size:
            self.dominated = self._select_by_crowding_distance(all_dominated, min(self.max_size - len(self.front), len(all_dominated)))

    def _get_non_dominated_solutions(self, solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
        """Get non-dominated solutions using NSGA-II style sorting."""
        if len(solutions) <= 1:
            return solutions, []
        return self._fast_non_dominated_sort(solutions)

    def _fast_non_dominated_sort(self, solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
        """Fast non-dominated sorting algorithm."""
        n = len(solutions)
        if n <= 1:
            return solutions, []
        
        # Initialize dominance structures
        domination_count = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
        
        # Calculate dominance relationships - O(N²M)
        for i in range(n):
            for j in range(i + 1, n):
                if solutions[i].dominates(solutions[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif solutions[j].dominates(solutions[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Find non-dominated solutions (domination_count = 0)
        non_dominated = []
        dominated = []
        
        for i in range(n):
            if domination_count[i] == 0:
                non_dominated.append(solutions[i])
            else:
                dominated.append(solutions[i])
        
        return non_dominated, dominated

    def _select_by_crowding_distance(self, solutions: list[Solution], k: int) -> list[Solution]:
        """Select solutions by crowding distance."""
        self.assign_crowding_distance(solutions)
        solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
        return solutions[:k]

    def _assign_crowding_distance(self, solutions: list[Solution]) -> None:
        """Assign crowding distance to solutions."""
        num_solutions = len(solutions)
        if num_solutions == 0:
            return

        for s in solutions:
            s.crowding_distance = 0

        for i in range(len(solutions[0].score)):
            solutions.sort(key=lambda s: s.score[i])
            solutions[0].crowding_distance = float("inf")
            solutions[-1].crowding_distance = float("inf")

            max_score = solutions[-1].score[i]
            min_score = solutions[0].score[i]
            if max_score == min_score:
                continue  # Skip this objective if all scores are the same

            for j in range(1, num_solutions - 1):
                if solutions[j + 1].score[i] != solutions[j - 1].score[i]:
                    solutions[j].crowding_distance += (
                        solutions[j + 1].score[i] - solutions[j - 1].score[i]
                    ) / (max_score - min_score)
                else:
                    solutions[j].crowding_distance += 0

    def __repr__(self):
        return f"Archive(front_size={len(self.front)}, dominated_size={len(self.dominated)})" 