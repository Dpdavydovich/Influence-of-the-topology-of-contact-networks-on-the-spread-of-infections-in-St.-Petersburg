import numpy as np
import pandas as pd
import os
import time
# import json
# from tqdm.notebook import tqdm
# import json
# from IPython.display import display
# import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
from collections import defaultdict
import pickle
import csv
from igraph import *
# warnings.filterwarnings('ignore')

pop_folder = 'generated_pops2'
pop_folder_random = 'generated_pops2_random'
plot_folder = os.path.join('Plots', 'Degree_distributions')

def employed_status(age, sex):
    status = 0 # для детей <7 пенсионеров >65(M) или >60(F)
    if (sex == 'F') and (age > 17) and (age < 60): # для работающих женщин
        status = 2
    elif (sex == 'M') and (age > 17) and (age < 65): # для работающих мужчин
        status = 2
    elif (age >= 7) and (age <= 17):
        status = 1 # для учащихся
    return status

def plot_degree_distributions(closest_graph, random_graph, year):
    #range_max = max(max(closest_data), max(random_data)) + 100
    plt.figure(figsize=(15, 8))
    plt.title(f'Graph degree distributions for {year} year')
    plt.hist(closest_graph.degree(), bins=100, log=True, alpha=0.5, label='closest')#, color='b');
    plt.hist(random_graph.degree(), bins=100, log=True, alpha=0.5, label='random')#, color='peru');
    plt.xlabel('Degree')
    plt.ylabel('Amount')
    plt.legend()
    plt.savefig(os.path.join(plot_folder, f'{year}.png'))
    
def read_pop_from_csv(path_to_data):
    nodes = set()
    edges = []
    with open(os.path.join(path_to_data, f'people_{year}.txt')) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            person_id = row['sp_id']
            apart_id = 'a' + row['sp_hh_id']
            age = int(row['age'])
            sex = row['sex']
            work_id = row['work_id']
            status = employed_status(age, sex)

            nodes.add(person_id)
            edges.append((person_id, apart_id))

            if apart_id not in nodes:
                nodes.add(apart_id)

            if work_id.isdigit():
                if status == 1:
                    work_id = 's' + work_id
                elif status == 2:
                    work_id = 'w' + work_id
                if work_id not in nodes:
                    nodes.add(work_id)
                edges.append((person_id, work_id))
                
            if i%1000==0:
                print(f'{i}', end='\r')
        print()
        return nodes, edges

def create_igraph(nodes, edges):
    print('Create graph...')
    G = Graph()
    nodes = list(nodes)
    G.add_vertices(nodes)
    G.add_edges(edges)
    return G

def calc_graph_metrics(G):
    nodes_len, edges_len = len(G.vs), len(G.es)
    print(f'Nodes len: {nodes_len}\tEdges len: {edges_len}')
    avg_node_degree = sum(G.degree())/len(G.degree())
    print('Avg node degree:', avg_node_degree)
    density = G.density()
    print('Density:', density)
    degree_max = G.maxdegree()
    print(f'Max degree: {G.vs.select(_degree=G.maxdegree())["name"]} = {degree_max}')
    connected_components = G.clusters()._len
    print(f'Connected components: {connected_components}')
#     print('Edge connectivity', G.edge_connectivity())
#     print('Node connectivity', G.vertex_connectivity())
#    print('Graph is connected:', G.is_connected())

    return nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components

def save_metrics(container, nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components):
    container['nodes_len'].append(nodes_len)
    container['edges_len'].append(edges_len)
    container['avg_node_degree'].append(avg_node_degree)
    container['density'].append(density)
    container['degree_max'].append(degree_max)
    container['connected_components'].append(connected_components)

closest_metrics = defaultdict(list)
random_metrics = defaultdict(list)

for year in range(2010, 2020):
    start = time.time()
    
    print(f'YEAR == {year}')
    
    print('CLOSEST DATA')
    nodes, edges = read_pop_from_csv(pop_folder)
    G_closest = create_igraph(nodes, edges)
    nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components = calc_graph_metrics(G_closest)
    save_metrics(closest_metrics, nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components)
    
    print('RANDOM DATA')
    
    nodes, edges = read_pop_from_csv(pop_folder_random)
    G_random = create_igraph(nodes, edges)
    nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components = calc_graph_metrics(G_random)    
    
    save_metrics(random_metrics, nodes_len, edges_len, avg_node_degree, density, degree_max, connected_components)
    
    plot_degree_distributions(G_closest, G_random, year)
    end = time.time()
    print(f'Execution time for {year}: {end-start}s')

del G_closest, G_random

fig, axes = plt.subplots(len(closest_metrics), figsize=(10, 15), sharex=True, constrained_layout=True)
#fig.tight_layout()
years = list(range(2010, 2020))
for ax, metric in zip(axes, closest_metrics):
    ax.set_title(metric)
    ax.plot(years, closest_metrics[metric], label='closest', lw=5)
    ax.plot(years, random_metrics[metric], label = 'random', lw=5)
    ax.set_xticks(years, years)
    ax.grid()
    ax.legend();
    
plt.savefig('Plots/metrics.png')

