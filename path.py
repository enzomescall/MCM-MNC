import numpy as np

"""
Creating path object
"""

class Path:
    def __init__(self, path, heatmap, sonar_range, sonar_falloff):
        self.path = path
        self.heatmap = heatmap
        self.sonar_range = sonar_range
        self.sonar_falloff = sonar_falloff
        self.path_heat = self.test_path(path, heatmap, sonar_range, sonar_falloff)

    # Function to check if coordinates are within bounds
    def is_within_bounds(self, x, y, heatmap):
        if x < 0 or x >= heatmap.shape[0] or y < 0 or y >= heatmap.shape[1]:
            return False
        return True

    # Function to test a path
    def test_path(self, path, heatmap, sonar_range, sonar_falloff):
        # Deep copy heatmap to avoid modifying the original
        heatmap = heatmap.copy()

        path_heat = 0
        for i in range(len(path)):
            x, y = path[i]
            path_heat += self.heat(x, y, heatmap, sonar_range, sonar_falloff)
        return path_heat

    def heat(self, x, y, heatmap, sonar_range, sonar_falloff):
        heat = 0

        if not self.is_within_bounds(x, y, heatmap):
            print("Path out of bounds")
            return 0
        
        heat = heatmap[x, y]
        heatmap[x, y] = 0  # Remove heat from the visited position

        # Add the heat from the sonar
        for i in range(1, sonar_range + 1):
            for dx in range(-i, i+1):
                for dy in range(-i, i+1):
                    nx = x + dx
                    ny = y + dy
                    if self.is_within_bounds(nx, ny, heatmap):
                        distance = ((dx ** 2) + (dy ** 2)) ** 0.5  # Euclidean distance
                        sonar_heat = heatmap[nx, ny] * sonar_falloff ** distance
                        heatmap[nx, ny] -= sonar_heat
                        heat += sonar_heat

        return heat

    def __lt__(self, other):
        return self.path_heat < other.path_heat

    def __eq__(self, other):
        return self.path_heat == other.path_heat

    def __gt__(self, other):
        return self.path_heat > other.path_heat