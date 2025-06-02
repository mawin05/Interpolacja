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

def get_chebyshev_nodes(dist, n):
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))

    indices = []
    for val in x_cheb:
        idx = int((np.abs(dist - val)).argmin())
        indices.append(idx)
    return indices


def lagrange(dist, height, n, chebyshev):
    dist = scale_distance(dist)
    nodes = []
    if chebyshev:
        nodes = get_chebyshev_nodes(dist, n)
    else:
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

def get_splines(dist, height, n, resolution_per_interval=10):
    dist = scale_distance(dist)
    nodes = get_nodes(dist, n)
    x = dist[nodes]
    y = height[nodes]

    h = x[1:] - x[:-1]

    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y[:-1].copy()

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    interpolated_y = []

    for xi in dist:
        i = np.searchsorted(x, xi)-1
        i = max(0, min(i, n-2))
        dx = xi - x[i]
        sx = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
        interpolated_y.append(sx)


    return interpolated_y, nodes


def interpolate(data, name, chebyshev, method):
    distance = np.array(data.iloc[:,0].tolist())
    height = np.array(data.iloc[:,1].tolist())

    number_of_nodes = [10, 20, 40, 60]

    for num in number_of_nodes:
        if method == "lagrange":
            interpolated_heights, nodes = lagrange(distance, height, num, chebyshev)
            method_name = "Lagrange'a"
            method_path = '/lagrange'
        else:
            interpolated_heights, nodes = get_splines(distance, height, num)
            method_name = "splajnów"
            method_path = "/spline"

        x_nodes = distance[nodes]
        y_nodes = height[nodes]
        plt.plot(distance, height, color='blue', label='Dane oryginalne-'+name)
        plt.plot(distance, interpolated_heights, color="red", label="Interpolacja metodą "+method_name)
        plt.plot(x_nodes, y_nodes, marker='o', linestyle='none', color="green", label="Węzły interpolacji")
        if name == 'MountEverest':
            plt.ylim(6000, 9000)
        else:
            plt.ylim(0, 3000)
        plt.xlabel('Dystans (m)')
        plt.ylabel('Wysokość (m)')
        plt.title("Metoda "+method_name+" - " + str(num) + " węzłów")
        plt.grid(True)
        plt.legend()
        path = 'wykresy/' + name
        if chebyshev:
            path += '/chebyshev'
        path += method_path + str(num) +'.png'
        plt.savefig(path)
        plt.close()

everest_data = pd.read_csv('2018_paths/MountEverest.csv', sep=',', skiprows=1, header=None)
colorado_data = pd.read_csv('2018_paths/WielkiKanionKolorado.csv', sep=',', skiprows=1, header=None)
# interpolate(everest_data, 'MountEverest', False)
# interpolate(colorado_data, 'WielkiKanion', False)
# interpolate(everest_data, "MountEverest", True)
interpolate(colorado_data, 'WielkiKanion', False, 'spline')

