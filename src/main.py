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

def interpolate(data, name):
    distance = np.array(data.iloc[:,0].tolist())
    height = np.array(data.iloc[:,1].tolist())

    number_of_nodes = [10, 20, 40, 80]

    for num in number_of_nodes:
        interpolated_heights, nodes = lagrange(distance, height, num)
        x_nodes = distance[nodes]
        y_nodes = height[nodes]
        plt.plot(distance, height, color='blue', label='Dane oryginalne-'+name)
        plt.plot(distance, interpolated_heights, color="red", label="Interpolacja metodą Lagrange'a")
        plt.plot(x_nodes, y_nodes, marker='o', linestyle='none', color="green", label="Węzły interpolacji")
        if name == 'MountEverest':
            plt.ylim(6000, 9000)
        else:
            plt.ylim(0, 3000)
        plt.xlabel('Dystans (m)')
        plt.ylabel('Wysokość (m)')
        plt.title("Metoda Lagrange'a - " + str(num) + " węzłów")
        plt.grid(True)
        plt.legend()
        plt.savefig('wykresy/' + name + '/lagrange' + str(num) +'.png')
        plt.close()

everest_data = pd.read_csv('2018_paths/MountEverest.csv', sep=',', skiprows=1, header=None)
colorado_data = pd.read_csv('2018_paths/WielkiKanionKolorado.csv', sep=',', skiprows=1, header=None)
interpolate(everest_data, 'MountEverest')
interpolate(colorado_data, 'WielkiKanion')
