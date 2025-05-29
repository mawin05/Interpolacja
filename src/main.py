import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scale_distance(dist):
    last_distance = dist[-1]
    first_distance = dist[0]
    return 2 * (dist - first_distance) / (last_distance - first_distance) - 1

def get_nodes(dist, n):
    indices = np.linspace(0, len(dist) - 1, n, dtype=int)
    return indices


def lagrange(dist, height, n):
    dist = scale_distance(dist)
    nodes = get_nodes(dist, n)
    y = []
    for point in range(0,len(dist)):
        x = dist[point]
        result = 0
        for i in range(0,n):
            xi = dist[nodes[i]]
            yi = height[nodes[i]]
            li = 1
            for j in range(0,n):
                if j!=i:
                    xj = dist[nodes[j]]
                    li *= (x-xj)/(xi-xj)
            result+=li*yi
        y.append(result)
    return np.array(y), nodes


everest_data = pd.read_csv('2018_paths/MountEverest.csv', sep=',', skiprows=1, header=None)
everest_dist = np.array(everest_data.iloc[:,0].tolist())
everest_height = np.array(everest_data.iloc[:,1].tolist())

y_interpolated, nodes = lagrange(everest_dist, everest_height, 80)
x_nodes = everest_dist[nodes]
y_nodes = y_interpolated[nodes]

plt.plot(everest_dist, everest_height, color='blue', label='Dane oryginalne')
plt.plot(everest_dist, y_interpolated, color="red", label="Interpolacja metodą Lagrange'a")
plt.plot(x_nodes, y_nodes, marker='o', linestyle='none', color="green", label="Węzły interpolacji")
plt.ylim(6000, 9000)
plt.xlabel('Dystans (m)')
plt.ylabel('Wysokość (m)')
plt.title("Metoda Lagrange'a - 80 węzłów")
plt.grid(True)
plt.legend()
plt.savefig('wykresy/MountEverest/lagrange80.png')
plt.show()
