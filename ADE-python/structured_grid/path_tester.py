from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from path_generator import generate_path, is_within_bounds, distance

# set seed
np.random.seed(0)

# Function to generate a heatmap based on an equation
def generate_heatmap(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    heatmap = -np.cos(3*x+6) - np.sin(3*y+2) - 0.5 * np.cos(5*x + 1) - 0.3*(y + 2)**2 + y - 0.3

    return -heatmap

@njit
def alter_heatmap(heatmap, path, sonar_range, sonar_falloff):
    total_altered = 0
    for x,y in path:
        for i in range(sonar_range):
            for dx in range(-i, i+1):
                for dy in range(-i, i+1):
                    nx = x + dx
                    ny = y + dy
                    if is_within_bounds(nx, ny, heatmap):
                        d = distance(x, y, nx, ny)
                        if d == 0:
                            total_altered += heatmap[nx, ny]
                            heatmap[nx, ny] = 0
                            continue
                        sonar_heat = heatmap[nx, ny] / (d ** sonar_falloff)
                        heatmap[nx, ny] -= sonar_heat
                        total_altered += sonar_heat

    #print("Heatmap altered, sonar range:", sonar_range, "sonar falloff:", sonar_falloff, "total heat altered:", total_altered)

def lt(x):
    return x*0.1 + 37

def lg(x):
    return x*0.1 + 17

def complete_search(k_generations, n_paths, start_point, desired_path_length, heatmap, sonar_range, sonar_falloff):
    # deep copy heatmap
    heatmap = heatmap.copy()

    top_path = (None, heat := -np.inf)
    path = [start_point]
    for i in range(k_generations):
        print(f"k = {i+1}/{k_generations}")

        newpath, fullpath, newheat = generate_path(n_paths, path, desired_path_length, heatmap, sonar_range, sonar_falloff)

        print(f"Total path heat: {newheat}, total nodes visited: {len(newpath)}")

        if newheat > 0 or newheat > heat:
            path = fullpath
            heat = newheat
            top_path = (path, heat)
        else:
            print("No better path found, stopping search")
            return top_path

        alter_heatmap(heatmap, newpath, sonar_range, sonar_falloff)

    return top_path[0], top_path[1], heatmap

# Your existing code...

def stepwise_search(k_generations, n_paths, start_point, desired_path_length, heatmap_list, sonar_range, sonar_falloff, heatmap_skip = 20):
    top_path = (None, heat := -np.inf)
    path = [start_point]

    for i, heatmap in enumerate(heatmap_list):
        if i % heatmap_skip != 0:
            # skip this heatmap
            continue

        print(f'Searching heatmap {i+1}/{len(heatmap_list)}')
        # deep copy heatmap
        heatmap = heatmap.copy()

        # if i > 500:
        #     desired_path_length = 40
        #     sonar_range = 5

        for i in range(k_generations):
            print(f"k = {i+1}/{k_generations}")

            newpath, fullpath, newheat = generate_path(n_paths, path, desired_path_length, heatmap, sonar_range, sonar_falloff)

            print(f"Total path heat: {newheat}, total nodes visited: {len(newpath)}")

            if newheat > heat:
                path = fullpath
                heat = newheat
                top_path = (path, heat)

            alter_heatmap(heatmap, newpath, sonar_range, sonar_falloff)

        # ready to move to next heatmap
        heat = -np.inf

    return top_path[0], top_path[1], heatmap

def check_search(k_generations, n_paths, start_point, desired_path_length, heatmap_list, sonar_range, sonar_falloff, submersible_paths, heatmap_skip = 20):   
    path = [start_point]
    subs_found = 0
    heat = -np.inf
    num_submersibles = len(submersible_paths)

    for j, heatmap in enumerate(heatmap_list):
        if j % 10 == 0:
            print(f'Searching heatmap {j+1}/{len(heatmap_list)}')

        if j % heatmap_skip != 0:
            # skip this heatmap
            continue
        # deep copy heatmap
        heatmap = heatmap.copy()

        for i in range(k_generations):
            newpath, fullpath, newheat = generate_path(n_paths, path, desired_path_length, heatmap, sonar_range, sonar_falloff)
            
            for i,_ in enumerate(submersible_paths):
                submersible = submersible_paths.pop(0)
                dist_to_sub = distance(fullpath[-1][0], fullpath[-1][1], submersible[j][0], submersible[j][1])
                
                if  dist_to_sub < sonar_range * np.random.uniform(0.5, 1.5):
                    print("Submersible found!")
                    subs_found += 1
                    if subs_found == num_submersibles:
                        print(f'All {num_submersibles} submersibles found in {j} timestamps')
                        return subs_found, j
                else:
                    submersible_paths.append(submersible)
            
            if newheat > heat:
                path = fullpath
                heat = newheat

            alter_heatmap(heatmap, newpath, sonar_range, sonar_falloff)

        # ready to move to next heatmap
        heat = -np.inf

    return subs_found, None

if __name__ == "__main__":
    # Generate heatmap
    heatmap_shape = (50, 50)
    heatmap = generate_heatmap(heatmap_shape)

    start_point = (12, 8)
    sonar_range = 3
    sonar_falloff = 2 # distance exponent factor

    top_path, heat, new_heatmap = complete_search(1, 1000, start_point, 100, heatmap, sonar_range, sonar_falloff)

    print(f"Top path heat: {heat}, total nodes visited: {len(top_path)}")

    plt.figure(figsize=(10, 4.5))

    heatmap = heatmap.T
    new_heatmap = new_heatmap.T

    # Plot the first subplot
    plt.subplot(1, 2, 1)
    plt.xlabel('Long. (X)')
    plt.ylabel('Lat. (Y)')
    plt.title("Path on Original Sample Space", fontsize=13)
    plt.imshow(heatmap, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='black', marker='.', label='Initial position', s=300, zorder=3)
    plt.scatter(lg(top_path[-1][0]), lt(top_path[-1][1]), color='black', marker='X', label=f'Final position at: {len(top_path)} steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path], [lt(y) for x, y in top_path], color='red', label='Search path', linewidth=3, zorder=1)
    plt.legend()
    plt.xlim(17.5, 20.5)
    plt.ylim(37.5, 40.5)

    # Plot the second subplot
    plt.subplot(1, 2, 2)
    plt.xlabel('Long. (X)')
    plt.title("Path on Altered Sample Space", fontsize=13)
    plt.imshow(new_heatmap, interpolation='nearest', origin="lower", extent=(lg(0), lg(50), lt(0), lt(50)))
    plt.scatter(lg(start_point[0]), lt(start_point[1]), color='black', marker='.', label='Start Point', s=300, zorder=3)
    plt.scatter(lg(top_path[-1][0]), lt(top_path[-1][1]), color='black', marker='X', label=f'Final Point After: {len(top_path)} Steps', s=150, zorder=2)
    plt.plot([lg(x) for x, y in top_path], [lt(y) for x, y in top_path], color='red', label='Search Vessel Path', linewidth=3, zorder=1)
    plt.xlim(17.5, 20.5)
    plt.ylim(37.5, 40.5)

    plt.suptitle('Trajectory of Search Vessel on Heat-map of Submersible Potential Locations', fontsize=15)
    plt.xlabel('Long. (X)')
    plt.ylabel('Lat. (Y)')

    plt.savefig('./path_stuff/example_path_2.png')

