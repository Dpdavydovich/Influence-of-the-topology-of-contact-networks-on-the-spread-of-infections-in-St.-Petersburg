import pandas as pd
import numpy as np
import json
import csv
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from igraph import *
import networkx as nx
plt.style.use('ggplot')

data_folder = '../datasets_workplace_school'

with open(os.path.join(data_folder, 'apart_to_hh.json')) as f:
    apart_to_hh = json.load(f)
    
# with open(os.path.join(data_folder, 'geoid_to_workid.json')) as f:
#     geoid_to_workid = json.load(f)

school_coords = {}
with open(os.path.join(data_folder, 'schools.txt')) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        #print(row)
        school_coords['s' + row['sp_id']] = (float(row['longitude']), float(row['latitude']))

work_coords = {}
with open(os.path.join(data_folder, 'sp5-15km_last', 'workplaces_apart.txt')) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        #print(row)
        work_coords['w' + row['sp_id']] = (float(row['longitude']), float(row['latitude']))
        
hh_coords = {}
with open(os.path.join(data_folder, 'households.txt')) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        #print(row)
        hh_coords[row['sp_id']] = (float(row['longitude']), float(row['latitude']))
        
with open(os.path.join(data_folder, 'hh_schools_5km.json'), 'r') as f:
    hh_schools_5km = json.load(f, )
    
with open(os.path.join(data_folder, 'hh_works_15km_sorted.json'), 'r') as f:
    hh_works_15km = json.load(f, )

HOUSE_ID = '7255'
HOUSE_coords = hh_coords[HOUSE_ID]
print(HOUSE_coords)

HOUSE_coords[0] + radius

def employed_status(age, sex):
    status = 0 # для детей <7 пенсионеров >65(M) или >60(F)
    if (sex == 'F') and (age > 17) and (age < 60): # для работающих женщин
        status = 2
    elif (sex == 'M') and (age > 17) and (age < 65): # для работающих мужчин
        status = 2
    elif (age >= 7) and (age <= 17):
        status = 1 # для учащихся
    return status

def read_pop_from_csv(path_to_data, year, HOUSE_ID):
    nodes = [] #set()
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
            hh_id = apart_to_hh[apart_id[1:]]
            if hh_id == HOUSE_ID:
                nodes.append(person_id)
                edges.append((person_id, apart_id))

                if apart_id not in nodes:
                    nodes.append(apart_id)

                if work_id.isdigit():
                    if status == 1:
                        work_id = 's' + work_id
                    elif status == 2:
                        work_id = 'w' + work_id
                    if work_id not in nodes:
                        nodes.append(work_id)
                    edges.append((person_id, work_id))
                
            if i%1000==0:
                print(f'{i}', end='\r')
        closest_schools = ['s' + sch for sch in hh_schools_5km[HOUSE_ID] if sch not in nodes]
        nodes.extend(closest_schools)
        closest_works = ['w'+ work for work in  hh_works_15km[HOUSE_ID] if work not in nodes]
        nodes.extend(closest_works)
        print()
        return nodes, edges

def create_igraph(nodes, edges):
    print('Create igraph...')
    G = Graph()
    nodes = list(nodes)
    G.add_vertices(nodes)
    G.add_edges(edges)
    return G

def create_nxgraph(nodes, edges):
    print('Create nxgraph...')
    G = nx.Graph()
    nodes = list(nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def get_coords(lst):
    result = []
    for place in lst:
        result.append(pos_init[place])
    return result

def calc_coords(lst):
    lats = []
    lons = []
    for coords in lst:
        lats.append(coords[0])
        lons.append(coords[1])
    return (sum(lats)/len(lats), sum(lons)/len(lons))

"""### nxgraph"""

sizes = [3, 15, 10000]
np.clip(sizes, 10, 1000)

np.clip(5, 10, 1000)

def plot_graph(G, pos_init, year, folder_to_save):
    print('Plotting...')
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, 
                           k=1/100000,
                           pos = pos_init, 
                           fixed = [n for n in pos_init.keys() if n[0] in ['s', 'a', 'w']],
                           iterations = 200,
                           threshold=0.0001,
                           #scale=5,
                           center = HOUSE_coords,
                           seed=123
                          )
    print(max(pos_init.items(), key=lambda x: x[1][0]))
    print(max(pos_init.items(), key=lambda x: x[1][1]))
    
    colors={'s': 'red', 
            'w': 'limegreen',
            'a': 'cyan'
           }
    
    legend = {'blue': 'peoples',
              'red': 'schools', 
              'lime': 'works',
              'cyan': 'aparts'
             }
    
    nx.draw_networkx_edges(G, 
                           pos=pos, 
                           width=0.1,
#                            alpha = edges_alpha, 
#                            ax=ax, 
#                            edge_color=edge_color
                          )
    for k in colors.keys():
        nx.draw_networkx_nodes(G, 
                               pos=pos, 
                               nodelist=[n for n in G.nodes() if n.startswith(k)], 
                               node_size=[np.clip(np.power(G.degree()[n], 2.5), 20, 1000) for n in G.nodes() if n.startswith(k)],
                               node_color=colors[k], #with_labels=True,
                               edgecolors='white',
                               #label='aparts', 
                               alpha = 1, 
#                               ax=ax
                              )
    nx.draw_networkx_nodes(G, 
                           pos=pos, 
                           nodelist=[n for n in G.nodes() if n[0] not in colors.keys()], 
                           node_size=5,
                           node_color='blue', #with_labels=True,
                           #label='aparts', 
                           alpha = 0.1,
#                               ax=ax
                              )    
    
    
#     node_color = [colors.get(n[0]) if colors.get(n[0]) else 'blue' for n in G_closest.nodes]

#     sizes = [np.power(G_closest.degree()[n], 2) for n in G_closest.nodes()]
#     print(max(sizes))
#     sizes = np.clip(sizes, 10, 300)
#     print(sizes.max())
#     labels = {n: legend[c] for n, c in zip(G_closest.nodes, node_color)}
#     alphas = [1 if n[0] in ['s', 'a', 'w'] else 0.5 for n in G_closest.nodes()]
#     nx.draw(G_closest, 
#             pos=pos, 
#             node_size=sizes,
#             #with_labels=True,
#             width=0.1,
#             node_color=node_color, 
#             #labels=labels
#            )
    low_lat = min(pos.values(), key=lambda x: x[0])[0]
    low_lat = low_lat - 0.0005 * low_lat
    high_lat = max(pos.values(), key=lambda x: x[0])[0]
    high_lat = high_lat + 0.0005 * high_lat
    
    low_lon = min(pos.values(), key=lambda x: x[1])[1]
    low_lon = low_lon - 0.0005 * low_lon
    high_lon = max(pos.values(), key=lambda x: x[1])[1]
    high_lon = high_lon + 0.0005 * high_lon
    
    plt.xlim(low_lat, high_lat)
    plt.ylim(low_lon, high_lon)
    
#     plt.xlim(HOUSE_coords[0] - RADIUS_15, HOUSE_coords[0] + RADIUS_15)
#     plt.ylim(HOUSE_coords[1] - RADIUS_15, HOUSE_coords[1] + RADIUS_15)
    
    handles = []
    for c, label in legend.items():
        patch = mpatches.Patch(color=c, label=label)
        handles.append(patch)
    plt.legend(handles=handles)
    print('Saving plot...')
    plt.savefig(os.path.join(folder_to_save, f'house_{HOUSE_ID}_{year}.png'))

"""### Plot graph for closest group"""

pop_folder = 'generated_pops2'
plot_folder = os.path.join('Plots', 'Single_house_graphplot_closest')

house_radius = 0.001 # ~110m
years = np.arange(2012, 2020)
HOUSE_ID = '7255'
HOUSE_coords = hh_coords[HOUSE_ID]

RADIUS_15 = 15/11.1 * 0.1

for year in years:
    print(f'YEAR == {year}')
    nodes, edges = read_pop_from_csv(pop_folder, year, HOUSE_ID)
    G_closest = create_nxgraph(nodes, edges)
    
    pos_init= {}
    apart_nodes = [k for k in nodes if k.startswith('a')]
    apart_coords = {ap: (HOUSE_coords[0]+np.random.uniform(-house_radius, house_radius), \
                         HOUSE_coords[1]+np.random.uniform(-house_radius, house_radius)) for ap in apart_nodes}
    pos_init.update(work_coords)
    pos_init.update(school_coords)
    pos_init.update(apart_coords)
    
    person_coords = pd.DataFrame(edges, columns=['person', 'places']).groupby('person').agg(list)
    person_coords['places_coords'] = person_coords['places'].apply(get_coords)
    person_coords = person_coords.drop('places', axis=1)
    person_coords['person_coords'] = person_coords['places_coords'].apply(calc_coords)
    person_coords = person_coords.drop('places_coords', axis=1)
    person_coords = person_coords.to_dict('dict')['person_coords']
    
    pos_init.update(person_coords)
    pos_init = {k:v for k, v in pos_init.items() if k in G_closest.nodes}
    plot_graph(G_closest, pos_init, year, plot_folder)

"""### Plot graph for random group"""

pop_folder_random = 'generated_pops2_random'
plot_folder_random = os.path.join('Plots', 'Single_house_graphplot_random')

house_radius = 0.001 # ~110m
years = np.arange(2012, 2020)
HOUSE_ID = '7255'
HOUSE_coords = hh_coords[HOUSE_ID]

for year in years:
    print(f'YEAR == {year}')
    nodes, edges = read_pop_from_csv(pop_folder_random, year, HOUSE_ID)
    G_closest = create_nxgraph(nodes, edges)
    
    pos_init= {}
    apart_nodes = [k for k in nodes if k.startswith('a')]
    apart_coords = {ap: (HOUSE_coords[0]+np.random.uniform(-house_radius, house_radius), \
                         HOUSE_coords[1]+np.random.uniform(-house_radius, house_radius)) for ap in apart_nodes}
    #print(max(apart_coords.values(), key=lambda v: v[0]) )
    pos_init.update(work_coords)
    pos_init.update(school_coords)
    pos_init.update(apart_coords)
    
    person_coords = pd.DataFrame(edges, columns=['person', 'places']).groupby('person').agg(list)
    person_coords['places_coords'] = person_coords['places'].apply(get_coords)
    person_coords = person_coords.drop('places', axis=1)
    person_coords['person_coords'] = person_coords['places_coords'].apply(calc_coords)
    person_coords = person_coords.drop('places_coords', axis=1)
    person_coords = person_coords.to_dict('dict')['person_coords']
    
    pos_init.update(person_coords)
    pos_init = {k:v for k, v in pos_init.items() if k in G_closest.nodes}
    plot_graph(G_closest, pos_init, year, plot_folder_random)







pos = nx.spring_layout(G_closest, 
                           k=1/100000,
                           pos = pos_init, 
                           fixed = [n for n in pos_init.keys() if n[0] in ['s', 'a', 'w']],
                           iterations = 200,
                           threshold=0.0001,
                           #scale=5,
                           center = HOUSE_coords,
                           seed=123
                          )

plt.style.use('ggplot')
print('Plotting...')
plt.figure(figsize=(20, 15))

print(max(pos_init.items(), key=lambda x: x[1][0]))
print(max(pos_init.items(), key=lambda x: x[1][1]))
print(min(pos_init.items(), key=lambda x: x[1][0]))
print(min(pos_init.items(), key=lambda x: x[1][1]))

colors={'s': 'red', 
        'w': 'limegreen',
        'a': 'cyan'
       }

legend = {'blue': 'peoples',
          'red': 'schools', 
          'lime': 'works',
          'cyan': 'aparts'
         }

nx.draw_networkx_edges(G_closest, 
                       pos=pos, 
                       width=0.1,
#                            alpha = edges_alpha, 
#                            ax=ax, 
#                            edge_color=edge_color
                      )
for k in colors.keys():
    nx.draw_networkx_nodes(G_closest, 
                           pos=pos, 
                           nodelist=[n for n in G_closest.nodes() if n.startswith(k)], 
                           node_size=[np.clip(np.power(G_closest.degree()[n], 2.5), 20, 1000) for n in G_closest.nodes() if n.startswith(k)],
                           node_color=colors[k], #with_labels=True,
                           edgecolors='black',
                           #label='aparts', 
                           alpha = 1, 
#                               ax=ax
                          )
nx.draw_networkx_nodes(G_closest, 
                       pos=pos, 
                       nodelist=[n for n in G_closest.nodes() if n[0] not in colors.keys()], 
                       node_size=10,
                       node_color='blue', #with_labels=True,
                       #label='aparts', 
                       alpha = 0.5,
#                               ax=ax
                          )    


#     node_color = [colors.get(n[0]) if colors.get(n[0]) else 'blue' for n in G_closest.nodes]

#     sizes = [np.power(G_closest.degree()[n], 2) for n in G_closest.nodes()]
#     print(max(sizes))
#     sizes = np.clip(sizes, 10, 300)
#     print(sizes.max())
#     labels = {n: legend[c] for n, c in zip(G_closest.nodes, node_color)}
#     alphas = [1 if n[0] in ['s', 'a', 'w'] else 0.5 for n in G_closest.nodes()]
#     nx.draw(G_closest, 
#             pos=pos, 
#             node_size=sizes,
#             #with_labels=True,
#             width=0.1,
#             node_color=node_color, 
#             #labels=labels
#            )
low_lat = min(pos.values(), key=lambda x: x[0])[0]
low_lat = low_lat - 0.0005 * low_lat
high_lat = max(pos.values(), key=lambda x: x[0])[0]
high_lat = high_lat + 0.0005 * high_lat

low_lon = min(pos.values(), key=lambda x: x[1])[1]
low_lon = low_lon - 0.0005 * low_lon
high_lon = max(pos.values(), key=lambda x: x[1])[1]
high_lon = high_lon + 0.0005 * high_lon
# plt.xlim(min(HOUSE_coords[0] - RADIUS_15, low_lat), HOUSE_coords[0] + RADIUS_15)
# plt.ylim(HOUSE_coords[1] - RADIUS_15, HOUSE_coords[1] + RADIUS_15)
plt.xlim(low_lat, high_lat)
plt.ylim(low_lon, high_lon)

handles = []
for c, label in legend.items():
    patch = mpatches.Patch(color=c, label=label)
    handles.append(patch)
plt.legend(handles=handles)