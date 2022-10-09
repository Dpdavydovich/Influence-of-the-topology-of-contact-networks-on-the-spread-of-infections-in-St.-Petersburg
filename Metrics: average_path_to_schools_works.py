

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from functools import lru_cache

def employed_status(age, sex):
    status = 0 # для детей <7 пенсионеров >65(M) или >60(F)
    if (sex == 'F') and (age > 17) and (age < 60): # для работающих женщин
        status = 2
    elif (sex == 'M') and (age > 17) and (age < 65): # для работающих мужчин
        status = 2
    elif (age >= 7) and (age <= 17):
        status = 1 # для учащихся
    return status

data_folder = '../datasets_workplace_school'
pop_folder = 'generated_pops2'
pop_folder_random = 'generated_pops2_random'
plot_folder = os.path.join('Plots', 'Average_distances_plot_hists')
plot_folder

os.listdir(data_folder)

with open(os.path.join(data_folder, 'apart_to_hh.json')) as f:
    apart_to_hh = json.load(f)

with open(os.path.join(data_folder, 'geoid_to_workid.json')) as f:
    geoid_to_workid = json.load(f)
    
with open(os.path.join(data_folder, 'workid_to_geoid.json')) as f:
    workid_to_geoid = json.load(f)
    
hh_schools_distances = pd.read_csv(os.path.join(data_folder, 'hh_schools_distances.csv'), 
                                   index_col=0
                                  )
hh_works_distances = pd.read_csv(os.path.join(data_folder, 'hh_works_distances.csv'), 
                                 index_col=0
                                )

@lru_cache(maxsize=2000000)
def get_hhid(apart_id):
    return int(apart_to_hh[apart_id])

@lru_cache(maxsize=500000)
def get_dist(hh_id, work_id, employed_status):
    if employed_status==1:
        return hh_schools_distances.loc[hh_id, work_id]
    elif employed_status==2:
        geo_id = get_geoid(work_id)
        return hh_works_distances.loc[hh_id, geo_id]
    
@lru_cache(maxsize=500000)
def get_geoid(work_id):
    return workid_to_geoid[work_id]
    
def get_distances(data):
    distances = []
    for p in data:
        #print(p) {'sp_id': 2307978, 'sp_hh_id': '40218', 'age': 7, 'sex': 'F', 'factor_age': 1, 'work_id': '536'}
        hh_id = get_hhid(p['sp_hh_id'])
        d = get_dist(hh_id, p['work_id'], p['employed_status'])
        distances.append(d)
        
    avg_dist = sum(distances) / len(distances)
    return distances, avg_dist

def plot_distributions(closest_data, random_data, school_or_work, year):
    range_max = max(max(closest_data), max(random_data)) + 100
    plt.figure(figsize=(15, 8))
    plt.hist(closest_data, bins=200, range=(0, range_max), alpha=0.5, label='closest')
    plt.hist(random_data, bins=200, range=(0, range_max), alpha=0.5, label = 'random')
    plt.xlabel('Distance, m')
    plt.ylabel('People amount')
    plt.legend()
    plt.savefig(os.path.join(plot_folder, f'{school_or_work}_{year}.png'))

avg_schools_closest = []
avg_works_closest = []
avg_schools_random = []
avg_works_random = []

for year in range(2010, 2020):
    print(f'YEAR == {year}')
    input_file = f'people_{year}.txt'
    print('Read population csv...')
    pop_closest = pd.read_csv(os.path.join(pop_folder, input_file), 
                                  sep='\t', 
                                  index_col=0, 
                                  dtype={'sp_id': np.int64, 'sp_hh_id': str, 'age': np.int32, 'sex': str, 'work_id':str}
                                 )
    pop_random = pd.read_csv(os.path.join(pop_folder_random, input_file), 
                                 sep='\t', 
                                 index_col=0, 
                                 dtype={'sp_id': np.int64, 'sp_hh_id': str, 'age': np.int32, 'sex': str, 'work_id':str}
                                )
    
    print('Deleting unemployed and set status...')
    pop_closest = pop_closest[pop_closest['work_id']!='X']
    pop_closest['employed_status'] = pop_closest.apply(lambda row: employed_status(row['age'], row['sex']), axis=1)
    pop_random = pop_random[pop_random['work_id']!='X']
    pop_random['employed_status'] = pop_random.apply(lambda row: employed_status(row['age'], row['sex']), axis=1)
    
    print('Create students and workers...')
    students_closest = pop_closest[(pop_closest['age']>=7)&(pop_closest['age']<=17)].to_dict('records')
    workers_closest = pop_closest[pop_closest['age'] > 17].to_dict('records')
    
    students_random = pop_random[(pop_random['age']>=7)&(pop_random['age']<=17)].to_dict('records')
    workers_random = pop_random[pop_random['age'] > 17].to_dict('records')
    
    print('Calc dists...')
    schools_closest_dists, avg_to_closest_school_dist = get_distances(students_closest)
    works_closest_dists, avg_to_closest_work_dist = get_distances(workers_closest)
    
    schools_random_dists, avg_to_random_school_dist = get_distances(students_random)
    works_random_dists, avg_to_random_work_dist = get_distances(workers_random)
    
    avg_schools_closest.append(avg_to_closest_school_dist)
    avg_works_closest.append(avg_to_closest_work_dist)
    avg_schools_random.append(avg_to_random_school_dist)
    avg_works_random.append(avg_to_random_work_dist)
    
    print('Plotting and saving...')
    plot_distributions(schools_closest_dists, schools_random_dists, 'schools', year)
    plot_distributions(works_closest_dists, works_random_dists, 'works', year)

years = list(range(2010, 2020))
plt.figure(figsize=(10, 6))
plt.title('Average distance from household to school')
plt.plot(years, avg_schools_closest, label = 'closest', lw=5)
plt.plot(years, avg_schools_random, label = 'random', lw=5)
plt.xlabel('Year')
plt.ylabel('Average distance')
plt.legend()
plt.savefig(os.path.join(plot_folder, 'schools_average_dists2010-2019.png'))

plt.figure(figsize=(10, 6))
plt.title('Average distance from household to work')
plt.plot(years, avg_works_closest, label = 'closest', lw=5)
plt.plot(years, avg_works_random, label = 'random', lw=5)
plt.xlabel('Year')
plt.ylabel('Average distance')
plt.legend()
plt.savefig(os.path.join(plot_folder, 'works_average_dists2010-2019.png'))

"""### Create JSON "workid_to_geoid" from JSON "geoid_to_workid"
"""

workid_to_geoid={work_id: geoid for i, geoid, work_id in pd.DataFrame(geoid_to_workid.items(), 
                                                            columns=['geoid', 'work_id']).explode('work_id').itertuples()}

with open(os.path.join(data_folder, 'workid_to_geoid.json'), 'w') as f:
    json.dump(workid_to_geoid, f)

pd.DataFrame({'to_school':[avg_to_closest_school_dist, avg_to_random_school_dist], 
              'to_work': [avg_to_closest_work_dist, avg_to_random_work_dist]}, 
              index=['closest', 'random']
             )

plt.figure(figsize=(15, 8))
plt.hist(schools_closest_dists, bins=200, range=(0, 15000), alpha=0.5, label='closest')
plt.hist(schools_random_dists, bins=200, range=(0, 15000), alpha=0.5, label = 'random')
plt.legend();

len(schools_closest_dists), len(schools_random_dists)

plt.figure(figsize=(15, 8))
plt.hist(works_closest_dists, bins=100, range=(0, 40000), alpha=0.5, label='closest')
plt.hist(works_random_dists, bins=100, range=(0, 40000), alpha=0.5, label = 'random')
plt.legend();

