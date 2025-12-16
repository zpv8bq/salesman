import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datareader import get_data

def cities_to_array(cities):
    coords = np.zeros((len(cities), 2))
    for i, c in enumerate(cities):
        coords[i] = [c.lon, c.lat]
    return coords

@jit(nopython=True)
def haversine_distance(coords):
    R = 6371.0
    coords_rad = np.radians(coords)
    n = len(coords_rad)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        lat1 = coords_rad[i, 1]
        lon1 = coords_rad[i, 0]
        lats2 = coords_rad[:, 1]
        lons2 = coords_rad[:, 0]
        dlat = lats2 - lat1
        dlon = lons2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dist_matrix[i] = R * c
    return dist_matrix

@jit(nopython=True)
def total_distance(route, dist_matrix):
    dist = 0.0
    for i in range(len(route)):
        dist += dist_matrix[route[i], route[(i+1) % len(route)]]
    return dist

def reverse_segment(route):
    new_route = route.copy()
    i, j = sorted(np.random.randint(0, len(route), 2))
    if i != j:
        new_route[i:j+1] = new_route[i:j+1][::-1]
    return new_route

def swap_cities(route):
    new_route = route.copy()
    i, j = np.random.randint(0, len(route),2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def initial_temp(cities, dist_matrix, n_samples=100):
    n_cities = len(cities)
    route = np.random.permutation(n_cities)
    deltas = []
    for _ in range(n_samples):
        if np.random.random() < 0.5:
            new_route = reverse_segment(route)
        else:
            new_route = swap_cities(route)
        old_dist = total_distance(route, dist_matrix)
        new_dist = total_distance(new_route, dist_matrix)
        delta = abs(new_dist - old_dist)
        deltas.append(delta)
        route = new_route
    max_delta = np.max(deltas)
    T_init = 2 * max_delta
    return T_init

def simulated_annealing(cities, T_init=None, T_min=0.05, alpha=0.9995, max_iter_per_temp=100):
    n_cities = len(cities)
    coords = cities_to_array(cities)
    dist_matrix = haversine_distance(coords)
    if T_init is None:
        T_init = initial_temp(cities, dist_matrix)
    print("T_init = ", T_init)
    current_route = np.random.permutation(n_cities)
    current_dist = total_distance(current_route, dist_matrix)
    best_route = current_route.copy()
    best_dist = current_dist
    initial_dist = current_dist
    
    T = T_init
    history = []
  
    while T > T_min:
        for c in range(max_iter_per_temp):
            if np.random.random()<0.875:
                new_route = reverse_segment(current_route)
            else:
                new_route = swap_cities(current_route)
            new_dist = total_distance(new_route, dist_matrix)
            delta = new_dist - current_dist
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_route = new_route
                current_dist = new_dist                
                if current_dist < best_dist:
                    best_route = current_route.copy()
                    best_dist = current_dist        
        history.append((T, best_dist))        
        T *= alpha
    return best_route, best_dist, initial_dist, history

def save_route(cities, route, filename):
    with open(filename, 'w') as f:
        f.write("# longitude   latitude\n")
        for idx in route:
            c = cities[idx]
            f.write(f"{c.lon:.6f}  {c.lat:.6f}\n")

def plot_annealing_schedule(history, filename):
    temps, dists = zip(*history)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temps, dists)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Annealing Schedule: Distance vs Temperature")
    ax.set_xscale("log")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python tsp_solver.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    base_name = Path(input_file).stem
    cities = get_data(input_file)
    n_cities = len(cities)
    print(f"Loaded {n_cities} cities\n")
    if n_cities < 50:
        T_init, alpha, max_iter = None, 0.995, 100
    elif n_cities < 200:
        T_init, alpha, max_iter = None, 0.9998, 75
    else:
        T_init, alpha, max_iter = None, 0.9999, 50

    start_time = time.time()
    best_route, best_dist, initial_dist, history = simulated_annealing(cities,T_init, 0.05,alpha,max_iter)
    elapsed = time.time() - start_time
    output_file = f"{base_name}_solution.dat"
    save_route(cities, best_route, output_file)   
    suffix = base_name.replace("cities", "")
    schedule_name = f"an{suffix}.png"
    plot_annealing_schedule(history, schedule_name)
    
    print(f"Initial distance:   {initial_dist:10.2f} km")
    print(f"Optimized distance: {best_dist:10.2f} km")
    print(f"Improvement:        {initial_dist - best_dist:10.2f} km "f"({100*(initial_dist-best_dist)/initial_dist:5.1f}%)")
    print(f"Execution time:     {elapsed:10.2f} seconds")

if __name__ == "__main__":
    main()
