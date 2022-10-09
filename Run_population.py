import numpy as np
import pandas as pd
import random
from collections import Counter
from population_classes_v2 import *
# import matplotlib.pyplot as plt
# import datetime
import pickle
from IPython.display import display
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

def factor_age(age):
    if age < 80:
        k = age // 5
    else:
        k = 16
    return k

def lagrange(xpl, x, y, n=2):
    arr1 = []
    for t, xt in enumerate(x):
        if xpl < xt:
            if len(x) - t < n:
                for j in range(t-n, t+1):
                    b = 1
                    for k in range(t-n, t+1):
                        if k != j:
                            b *= (xpl - x[k]) / (x[j] - x[k])
                    a = y[j] * b
                    arr1.append(a)
            else:
                for j in range(n+1):
                    b = 1
                    for k in range(n+1):
                        if k != j:
                            b *= (xpl - x[k+t-1]) / (x[j+t-1] - x[k+t-1])
                    a = y[j+t-1] * b
                    arr1.append(a)
            return sum(arr1)
        
def mape(real, simulated):
    return np.mean(np.abs(real-simulated) / real)


mortality = [61732, 62025, 60308, 60218, 61996, 61552, 60690, 59844, 59200]  # реальное количество умерших по годам (2011-2019)
mortality_male = np.array([[1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0, 0.9],  # смертность мужчин на 1000 человек соответствующего возраста (по 5 лет)
                           [0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1, 0.2],
                           [0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.2],
                           [0.5, 0.7, 0.6, 0.5, 0.7, 0.7, 0.8, 0.9, 0.9],
                           [1.2, 1.2, 1.1, 1.1, 0.9, 1.1, 1.0, 1.2, 1.3],
                           [2.7, 2.4, 2.1, 2.0, 1.9, 1.5, 1.4, 1.3, 1.2],
                           [5.5, 4.9, 4.8, 4.2, 3.8, 3.1, 2.7, 2.6, 2.2],
                           [6.3, 6.5, 6.2, 6.0, 5.9, 5.4, 4.9, 4.4, 4.2],
                           [7.0, 6.7, 6.5, 6.5, 6.9, 7.0, 6.6, 6.5, 6.4],
                           [9.5, 9.1, 7.9, 8.1, 7.9, 8.2, 7.9, 7.5, 7.4],
                           [13.9, 12.7, 11.5, 11.1, 11.7, 11.5, 11.5, 10.3, 10.1],
                           [20.3, 18.4, 16.9, 17.2, 16.7, 16.4, 15.6, 14.8, 14.1],
                           [29.2, 27.6, 25.8, 25.6, 25.6, 25.5, 23.6, 23.6, 22.8],
                           [36.8, 33.8, 33.3, 33.8, 34.5, 34.5, 33.7, 32.9, 29.4],
                           [73.3, 75.3, 73.6, 72.1, 73.2, 71.7, 68.2, 67.0, 65.8]]) / 1000


mortality_female = np.array([[1.4, 1.2, 1.2, 1.2, 1.2, 1.0, 0.9, 0.7, 0.7],  # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
                             [0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
                             [0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2],
                             [0.3, 0.3, 0.4, 0.4, 0.4, 0.3, 0.5, 0.4, 0.5],
                             [0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4],
                             [1.0, 0.8, 0.7, 0.7, 0.8, 0.6, 0.5, 0.5, 0.5],
                             [1.4, 1.6, 1.5, 1.3, 1.4, 1.3, 1.1, 0.9, 0.9],
                             [2.0, 2.2, 2.0, 1.9, 2.1, 1.8, 1.8, 1.8, 1.6],
                             [2.7, 2.5, 2.4, 2.5, 2.6, 2.6, 2.3, 2.3, 2.3],
                             [3.5, 3.6, 3.2, 3.2, 3.2, 3.4, 3.0, 3.0, 3.3],
                             [5.0, 4.9, 4.4, 4.5, 4.5, 4.2, 4.1, 4.1, 4.1],
                             [7.6, 6.8, 6.7, 6.5, 6.6, 6.4, 6.0, 5.8, 5.8],
                             [11.0, 10.8, 10.6, 9.7, 10.2, 9.3, 9.2, 9.0, 8.4],
                             [14.8, 15.1, 14.6, 15.0, 15.0, 15.2, 14.6, 13.7, 13.6],
                             [59.0, 61.4, 61.3, 62.1, 63.9, 62.3, 60.7, 57.8, 57.2]]) / 1000

rate_people = np.array([[ 4.2,  4.5,  4.9,  5.2,  5.6,  6. ,  6.3,  6.2,  6.1], # процент (доля) людей соответствующего возраста в популяции (по 5 лет) (2011-2019)
                       [ 3.8,  3.8,  3.8,  3.8,  3.9,  4. ,  4.3,  4.7,  5. ],
                       [ 3.4,  3.4,  3.4,  3.5,  3.6,  3.7,  3.7,  3.7,  3.8],
                       [ 5.1,  4.5,  4. ,  3.9,  3.7,  3.7,  3.8,  3.9,  4. ],
                       [ 8.8,  8.7,  8.3,  7.7,  6.9,  6. ,  5.2,  4.8,  4.6],
                       [ 8.8,  9.1,  9.4,  9.6,  9.7,  9.6,  9.3,  8.7,  7.9],
                       [ 8. ,  8.2,  8.4,  8.8,  9. ,  9.1,  9.3,  9.6,  9.7],
                       [ 7.4,  7.4,  7.5,  7.6,  7.7,  7.9,  8. ,  8.3,  8.6],
                       [ 6.7,  6.8,  7. ,  7.1,  7.2,  7.1,  7.2,  7.2,  7.3],
                       [ 7.5,  7.1,  6.7,  6.5,  6.4,  6.4,  6.5,  6.6,  6.8],
                       [ 8. ,  8. ,  7.8,  7.6,  7.4,  7. ,  6.6,  6.3,  6.1],
                       [ 7.1,  7.2,  7.2,  7.2,  7.2,  7.3,  7.3,  7.2,  7.1],
                       [ 6.5,  6.4,  6.3,  6.2,  6.2,  6.3,  6.4,  6.4,  6.6],
                       [ 3.1,  3.4,  4.3,  4.8,  5.3,  5.6,  5.5,  5.5,  5.5],
                       [11.6, 11.5, 11. , 10.5, 10.2, 10.3, 10.6, 10.9, 11.1]])/ 100


fertility = np.array([[11.9, 12.3, 11.9, 11.8, 11.3, 9.7, 8.2, 7.6, 7.],  # количество рожавших на 1000 женщин соответствующего возраста (по 5 лет с 15-ти) (2011-2019)
                      [53.0, 52.8, 50.4, 49.2, 51.1, 51.9, 48.2, 49.3, 47.7],
                      [98.7, 104.4, 102.9, 102.4, 106.4, 104.9, 92.7, 87.0, 76.7],
                      [73.3, 81.6, 83.5, 89.2, 94.4, 100.2, 91.7, 88.9, 81.9],
                      [34.7, 40.2, 42.2, 45.5, 47.9, 51.9, 49.8, 50.0, 48.6],
                      [7.3, 8.9, 9.0, 10.2, 11.3, 12.3, 12.3, 12.7, 12.],
                      [0.5, 0.4, 0.6, 0.7, 0.7, 1.0, 1.0, 1.1, 1.]]) / 1000

birth_sex = np.array([[29.5, 32.3, 33.2, 34.6, 36.5, 37.3, 34.3, 32.9, 30.3],  # количество родившихся мальчиков/девочек по годам
                      [27.5, 30.4, 31.0, 32.6, 34.2, 35.4, 32.2, 31.1, 28.6]]) * 1000

migration = np.array([[130321, 194511, 257636, 261972, 237084, 232663, 264780, 250742, 224852],  # реальное количество мигрировавших (им/э) по годам (прибыло/выбыло 2011-2019)
                      [71689, 120419, 157619, 209176, 211821, 187954, 200234, 222966, 210370]])

                            #2014,2017, 2013, 2018, 2019
migration_male = np.array([[4320, 6649, 4208, 6382, 6255, 4680, 7533, 3504, 8122, 6910],  # количество мигрировавших мужчин (3+/3-) соответствующего возраста (2010-2019) ВЫБЫЛО МУЖЧИН
                           [4270, 7335, 4159, 7510, 7796, 3594, 6631, 2691, 7463, 7383],
                           [2845, 4103, 2772, 4164, 4372, 2219, 3249, 1662, 3731, 3682],
                           [14226, 14193, 13857, 13562, 10827, 10205, 6145, 7642, 6916, 6394],
                           [23600, 15622, 24883, 14721, 11311, 20758, 13733, 13467, 17461, 17054],
                           [22702, 15935, 23910, 14223, 11317, 18520, 10917, 12978, 12283, 11121],
                           [17565, 16573, 17783, 15475, 13331, 14266, 12193, 10584, 13478, 12768],
                           [12719, 11778, 12888, 11215, 10476, 10885, 9101, 8075, 10325, 10269],
                           [10150, 8212, 10891, 8035, 7424, 8894, 6512, 6904, 7196, 7184],
                           [8287, 6491, 8386, 6187, 5644, 7377, 5352, 5726, 5779, 5579],
                           [6273, 5227, 5864, 4849, 4352, 5505, 4352, 4354, 4637, 4376],
                           [4220, 4647, 3615, 4465, 4146, 3561, 3748, 2803, 3971, 3986],
                           [2488, 3224, 2191, 3153, 3099, 2150, 2805, 1944, 2937, 2783],
                           [1452, 2030, 1264, 2057, 2034, 1402, 1744, 1226, 1762, 1802],
                           [717, 964, 621, 1011, 1174, 605, 750, 530, 923, 989],
                           [673, 819, 400, 651, 541, 560, 654, 490, 525, 406],
                           [482, 698, 444, 723, 722, 332, 506, 290, 525, 568]])


migration_female = np.array([[4048, 6063, 3769, 6051, 5743, 4363, 6851, 3227, 7690, 6383],  # количество мигрировавших женщин (3+/3-) соответствующего возраста (2010-2019) ВЫБЫЛО ЖЕНЩИН
                             [4046, 6995, 3767, 6880, 7220, 3298, 6210, 2440, 6848, 6890],
                             [2946, 3906, 2743, 3851, 4121, 2127, 2989, 1574, 3400, 3485],
                             [12288, 14314, 11440, 13854, 10950, 7683, 5535, 5683, 6065, 5497],
                             [17582, 15546, 17390, 14526, 12083, 12177, 14060, 9083, 16711, 16315],
                             [20858, 19321, 20715, 16588, 13825, 14554, 12975, 11112, 13553, 11510],
                             [15725, 19370, 15046, 18318, 16257, 11754, 14482, 8783, 15840, 14698],
                             [10794, 12731, 10328, 12478, 12025, 8720, 9935, 6516, 11112, 10790],
                             [8088, 8647, 8015, 8259, 7915, 6428, 6458, 5006, 7326, 7072],
                             [6063, 6605, 6009, 6184, 5960, 4957, 4855, 3860, 5501, 5240],
                             [5988, 6057, 5411, 5499, 4960, 4622, 4407, 3894, 4544, 4252],
                             [4955, 6131, 4267, 5870, 5351, 3973, 4609, 3355, 4898, 4341],
                             [3695, 4721, 3186, 4573, 4464, 2984, 3566, 2594, 3942, 3551],
                             [2488, 3483, 2233, 3468, 3327, 2034, 2706, 1899, 2770, 2695],
                             [1598, 1789, 1435, 1947, 2225, 1097, 1354, 1024, 1527, 1710],
                             [1868, 2145, 1677, 1708, 1338, 1389, 1590, 1297, 1355, 1000],
                             [1953, 2456, 2069, 2305, 2267, 1503, 1727, 1403, 1850, 1687]])
                                                     #2014,  2017,  2013,  2018,   2019
migr_e_m = np.mean(migration_male[:, 5:] / np.array([115513, 95925, 84869, 108034, 103254]), axis=1) #ВЫБЫЛО М 
migr_e_f = np.mean(migration_female[:, 5:] / np.array([93663, 104309, 72750, 114932, 107116]), axis=1) # Ж
migr_im_m = np.mean(migration_male[:, :5] / np.array([136989, 124500, 138136, 118383, 104821]), axis=1)#ПРИБЫЛО М
migr_im_f = np.mean(migration_female[:, :5] / np.array([124983, 140280, 119500, 132359, 120031]), axis=1)# Ж
migration_2010 = np.array([[747, 890, 734, 850], # Это 0,1,2,3 данные (2010 видимо) по 4 переменным ниже 
                           [745, 594, 736, 556],
                           [734, 479, 697, 439],
                           [4790, 641, 5299, 629],
                           [5598, 1548, 7049, 1523],
                           [4955, 1886, 6020, 2267],
                           [3240, 1797, 3435, 1717],
                           [2330, 1361, 2324, 1241],
                           [1827, 1247, 1846, 1053],
                           [1447, 1031, 1392, 805],
                           [1177, 977, 1562, 883],
                           [989, 698, 1305, 819],
                           [655, 516, 992, 662],
                           [541, 454, 1065, 792],
                           [263, 196, 634, 416],
                           [213, 177, 747, 514],
                           [186, 115, 853, 557]])
migr_e_m_2010 = migration_2010[:, 1] / np.array([14607])
migr_e_f_2010 = migration_2010[:, 3] / np.array([15723])
migr_im_m_2010 = migration_2010[:, 0] / np.array([30437])
migr_im_f_2010 = migration_2010[:, 2] / np.array([36690])
ages_mort = [k + 2 for k in range(0, 75, 5)]
ages_mort.append(100)
ages_fert = [k + 2 for k in range(10, 50, 5)]
ages_fert.append(52)
ages_mi = [k + 2 for k in range(0, 80, 5)]
ages_mi.extend([87, 100])

migr_male = np.zeros((17, 18), dtype=float)
migr_female = np.zeros((17, 18), dtype=float)
ages = [2010, 2013, 2014, 2017, 2018, 2019, 2020]
e_male = migration_male[:, 5:] / np.array([115513, 95925, 84869, 108034, 103254]) #14, 17, 13, 18, 19
e_female = migration_female[:, 5:] / np.array([93663, 104309, 72750, 114932, 107116])
i_male = migration_male[:, :5] / np.array([136989, 124500, 138136, 118383, 104821])
i_female = migration_female[:, :5] / np.array([124983, 140280, 119500, 132359, 120031])

# Аппроксимация с помощью полинома Лагрнжа доли каждой возрастной категории им/эм для м/ж для годов 2011-2019 зная значения в [2010, 2013, 
# 2014, 2017, 2018, 2019, 2020]
#
for i in range(17):
    migr = [migr_im_m_2010[i], i_male[i, 2], i_male[i, 0], i_male[i, 1], i_male[i, 3], i_male[i, 4], i_male[i, 4]]
#     print(migr_male[i, 10])
#     print(np.array([lagrange(j, ages, migr) for j in range(2011, 2020)]))
    migr_male[i, :9] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

    migr = [migr_im_f_2010[i], i_female[i, 2], i_female[i, 0], i_female[i, 1], i_female[i, 3], i_female[i, 4], i_female[i, 4]]
    migr_female[i, :9] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

    migr = [migr_e_m_2010[i], e_male[i, 2], e_male[i, 0], e_male[i, 1], e_male[i, 3], e_male[i, 4], e_male[i, 4]]
    migr_male[i, 9:] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

    migr = [migr_e_f_2010[i], e_female[i, 2], e_female[i, 0], e_female[i, 1], e_female[i, 3], e_female[i, 4], e_female[i, 4]]
    migr_female[i, 9:] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

sex_male = np.zeros((2, 9), dtype=float)
sex_female = np.zeros((2, 9), dtype=float)

# Аппроксимация с помощью полинома Лагрнжа доли м/ж среди им/эм для годов 2011-2019 зная 
# значения в [2010, 2013, 2014, 2017, 2018, 2019, 2020]
migr = [30437 / (30437 + 36690), 138136 / (138136 + 119500), 136989 / (136989 + 124983), 124500 / (124500 + 140280),
        118383 / (118383 + 132359), 104821 / (104821 + 120031), 104821 / (104821 + 120031)]
sex_male[0, :] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

migr = [36690 / (30437 + 36690), 119500 / (138136 + 119500), 124983 / (136989 + 124983), 140280 / (124500 + 140280),
        132359 / (118383 + 132359), 120031 / (104821 + 120031), 120031 / (104821 + 120031)]
sex_female[0, :] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

migr = [14607 / (14607 + 15723), 84869 / (84869 + 72750), 115513 / (115513 + 93663), 95925 / (95925 + 104309),
        108034 / (108034 + 114932), 103254 / (103254 + 107116), 103254 / (103254 + 107116)]
sex_male[1, :] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

migr = [15723 / (14607 + 15723), 72750 / (84869 + 72750), 93663 / (115513 + 93663), 104309 / (95925 + 104309),
        114932 / (108034 + 114932), 107116 / (103254 + 107116), 107116 / (103254 + 107116)]
sex_female[1, :] = np.array([lagrange(j, ages, migr) for j in range(2011, 2020)])

mortality_sim = []
fertility_real = []
fertility_sim = []
imigration_real = migration[0]
imigration_sim = []
emigration_real = migration[1]
emigration_sim = []

folder = r"/Users/daviddavydovic/PycharmProjects/Kurs,1week,urok1/Diplom.ITMO".replace('\\', '/')
start = time.time()
print('Reading JSONS...')
with open(os.path.join(folder, 'apart_to_hh.json'), 'r') as f:
    apart_to_hh = json.load(f)
    
    
with open(os.path.join(folder, 'hh_schools_5km.json'), 'r') as f:
    hh_schools_5km = json.load(f, )
    
with open(os.path.join(folder, 'hh_schools_5_15km.json'), 'r') as f:
    hh_schools_5_15km = json.load(f, )

with open(os.path.join(folder, 'hh_works_15km_sorted.json'), 'r') as f:
    hh_works_15km_sorted = json.load(f, )

with open(os.path.join(folder, 'geoid_to_workid.json'), 'r') as f:
    geoid_to_workid = json.load(f, )
    
# with open(os.path.join(folder, 'hh_works_5km.json'), 'r') as f:
#     hh_works_5km = json.load(f, )
    
# with open(os.path.join(folder, 'hh_works_5_15km.json'), 'r') as f:
#     hh_works_5_15km = json.load(f, )

print('Reading Buildings info...')
schools = read_building(os.path.join(folder, 'schools.txt'))
hh = read_building(os.path.join(folder, 'households.txt'))
aparts = read_building(os.path.join(folder, 'households_apart.txt'))
workplaces = read_building(os.path.join(folder, 'workplaces_apart.txt'))
empty_aparts = set(aparts['sp_id'].tolist())
empty_workplaces = set(workplaces['sp_id'].tolist())


population_folder = r"/Users/daviddavydovic/PycharmProjects/Kurs,1week,urok1/Diplom.ITMO/generated_pops2_random".replace('\\', '/')

print('Reading initial population...')
data = pd.read_csv(os.path.join(population_folder, 'people_2010_init.txt'), 
                sep='\t', 
                index_col=0, 
                dtype={'sp_id': np.int64, 'sp_hh_id': str, 'age': np.int32, 'sex': str, 'work_id':str})

#data = data.sample(1000000, random_state=12)
data_aparts = data['sp_hh_id'].unique()
data_workids = data['work_id'].unique()
#aparts = aparts[aparts['sp_id'].isin(data_aparts)]
#empty_aparts = set(aparts['sp_id'].tolist()) # False
print(f'Aparts count:{aparts.shape[0]}')
#workplaces = workplaces[workplaces['sp_id'].isin(data_workids)]
print(f'Workplaces count: {workplaces.shape[0]}')

init_pop_folder = r"/Users/daviddavydovic/PycharmProjects/Kurs,1week,urok1/Diplom.ITMO/initial_population_random".replace('\\', '/')
init_pop_file = 'init_pop.pickle'
if not os.path.exists(os.path.join(init_pop_folder, init_pop_file)):

    Schools = create_objects_dict(schools, School)

    WorkPlaces = create_objects_dict(workplaces, WorkPlace)

    Aparts = create_objects_dict(aparts, Apart)

    HouseHolds = create_objects_dict(hh, HouseHold)

    print(data.head())


    population = Population(Schools, WorkPlaces, 
                            Aparts, HouseHolds, apart_to_hh, 
                            empty_aparts, empty_workplaces, 
                            hh_schools_5km, hh_schools_5_15km, 
                            hh_works_15km_sorted, geoid_to_workid, 
                            #mode='random'
                           )
    population.create_population_from_csv(data) # !!! mode = ['closest', 'random']
    with open(os.path.join(init_pop_folder, 'init_pop.pickle'), 'wb') as f:
        pickle.dump(population, f)
else:
    print('Load population from file.')
    with open(os.path.join(init_pop_folder, init_pop_file), 'rb') as f:
        population = pickle.load(f)

pop2010 = population.create_dataframe()
pop2010.to_csv(os.path.join(population_folder, 'people_{}.txt'.format(2010)), sep='\t')

print(len(population.population), 'изначально')
size = len(population.population)

for year in range(2011, 2020):
    start_year = time.time()
    
    print('YEAR == ', year)

    population.update_year()# !!! mode = ['closest', 'random']

    data = population.create_dataframe()

    print(data.shape)
    print(data.head())

    # mortality
    numb_index_m, numb_index_f = [], []
    for i in range(15):
        if i < 14:
            curr = data[data.factor_age == i]
            curr_m = curr[curr.sex == 'M']
            curr_f = curr[curr.sex == 'F']
        else:
            curr = data[data.factor_age >= i]
            curr_m = curr[curr.sex == 'M']
            curr_f = curr[curr.sex == 'F']
        # numb_m = len(data) * rate_people[i, year - 2011] * len(curr_m) / len(curr) * mortality_male[i, year - 2011]
        # numb_f = len(data) * rate_people[i, year - 2011] * len(curr_f) / len(curr) * mortality_female[i, year - 2011]
        numb_m = len(curr_m) * mortality_male[i, year - 2011]
        numb_f = len(curr_f) * mortality_female[i, year - 2011]
        if i < 14:
            numb_index_m.append(numb_m / 5)
            numb_index_f.append(numb_f / 5)
            

# Симулируем количство умерших для возрастов >75, зная сколько было в предыдущие(ages_mort-середины возрастных категорий). Главное чтобы это # количество смертей  не превышало общее значение.
# 
    k = numb_m
    mort = 0
    for i in range(int(numb_m / 30), int(numb_m)):
        mort_m = numb_index_m.copy()
        mort_m.append(i)
        mort_m.append(0)
        temp_m = [lagrange(j, ages_mort, mort_m) for j in range(70, 100)]
        temp_m = [d if d > 0 else 0 for d in temp_m]
        if k > abs(numb_m - sum(temp_m)):
            k = abs(numb_m - sum(temp_m))
            mort = i
    numb_index_m.append(mort)
    numb_index_m.append(0)

    k = numb_f
    mort = 0
    for i in range(int(numb_f / 30), int(numb_f)):
        mort_f = numb_index_f.copy()
        mort_f.append(i)
        mort_f.append(0)
        temp_f = [lagrange(j, ages_mort, mort_f) for j in range(70, 100)]
        temp_f = [d if d > 0 else 0 for d in temp_f]
        if k > abs(numb_f - sum(temp_f)):
            k = abs(numb_f - sum(temp_f))
            mort = i
    numb_index_f.append(mort)
    numb_index_f.append(0)

    temp_m = [lagrange(j, ages_mort, numb_index_m) for j in range(100)]
    temp_m = [i if i > 0 else 0 for i in temp_m]
    temp_f = [lagrange(j, ages_mort, numb_index_f) for j in range(100)]
    temp_f = [i if i > 0 else 0 for i in temp_f]

    # plt.plot(range(100), [x + y for x, y in zip(temp_m, temp_f)], color='green')
    # plt.plot(range(100), temp_m, color='blue')
    # plt.plot(range(100), temp_f, color='red')
    # plt.show()

    curr_index = []
    for i in range(1, 100):
        curr = data[data.age == i]
        curr_m = curr[curr.sex == 'M']
        curr_f = curr[curr.sex == 'F']
        if len(curr_m) < int(round(temp_m[i])):
            temp_m[i] = len(curr_m)
        real_mort = np.random.choice(np.array(curr_m.sp_id), int(round(temp_m[i])), replace=False)
        curr_index.extend(real_mort)
        if len(curr_f) < int(round(temp_f[i])):
            temp_f[i] = len(curr_f)
        real_mort = np.random.choice(np.array(curr_f.sp_id), int(round(temp_f[i])), replace=False)
        curr_index.extend(real_mort)

    print(Counter(data.loc[data.sp_id.isin(curr_index), 'factor_age']))
    data = data[~data.sp_id.isin(curr_index)]

    print(f'len data:{data.shape}')

    population.mortality_emigration(curr_index)
    print(f'len population: {len(population.population)}')

    mortality_sim.append(size - len(data))
    
    print(len(data), 'после смертности', size - len(data))
    size = len(data)

    # fertility
    numb_f_all = 0
    numb_index = [0]
    for i in range(3, 10):
        curr = data[(data.factor_age == i) & (data.sex == 'F')]
        numb_f = len(curr) * fertility[i - 3, year - 2011]
        numb_f_all += numb_f
        numb_index.append(numb_f / 5)
        
    fertility_real.append(numb_f_all) # общее количество фертильных женщин

    numb_index.append(0)
    temp = [lagrange(j, ages_fert, numb_index) for j in range(15, 52)]
    temp = [i if i > 0 else 0 for i in temp]

    temp = [np.sum(birth_sex, axis=0)[year - 2011] * i / sum(temp) for i in temp]
    curr_index = []
    for i in range(15, 52):
        curr = data[(data.age == i) & (data.sex == 'F')]
        #print(len(curr.sp_id), int(round(temp[i - 15])))
        #decrement = 0.5 * len(curr.sp_id) # !!! REMOVE !!!
        real_fert = np.random.choice(np.array(curr.sp_id), int(round(temp[i - 15])), replace=False) # !!!!!!!!
        curr_index.extend(real_fert)

    hh_list = list(data[data.sp_id.isin(curr_index)]['sp_hh_id'])
    if len(hh_list) < birth_sex.sum(axis=0)[year - 2011]:
        numb_f = int(birth_sex.sum(axis=0)[year - 2011] - len(hh_list))
        real_fert = np.random.choice(np.array(hh_list), numb_f, replace=False) # True
        #real_fert = np.random.choice(np.array(hh_list), 100, replace=False) # False
        hh_list.extend(real_fert)

    real_fert_m = np.random.choice(np.array(hh_list), int(birth_sex[0, year - 2011]), replace=False) # True
    #real_fert_m = np.random.choice(np.array(hh_list), 50, replace=False) # False
    count = Counter(hh_list)
    count_1 = Counter(real_fert_m)
    real_fert_f = list((count - count_1).elements())

    id_list = list(range(max(data.sp_id) + 1, max(data.sp_id) + len(real_fert_m) + 1))
    age_list = [0] * len(real_fert_m)
    sex_list = ['M'] * len(real_fert_m)
    new_frame = pd.DataFrame({'sp_id': id_list, 'sp_hh_id': real_fert_m, 'age': age_list, 'sex': sex_list, 'factor_age': age_list})
    data = data.append(new_frame, ignore_index=True)
    population.fertility(new_frame)
    print(Counter(new_frame.sex))

    id_list = list(range(max(data.sp_id) + 1, max(data.sp_id) + len(real_fert_f) + 1))
    age_list = [0] * len(real_fert_f)
    sex_list = ['F'] * len(real_fert_f)
    new_frame = pd.DataFrame({'sp_id': id_list, 'sp_hh_id': real_fert_f, 'age': age_list, 'sex': sex_list, 'factor_age': age_list})
    data = data.append(new_frame, ignore_index=True)
    data = data.fillna(0)

    population.fertility(new_frame)

    print(Counter(new_frame.sex))
    
    fertility_sim.append(len(data) - size)

    print(len(data), 'после рождаемости', len(data) - size)
    size = len(data)
    print(f'len population: {len(population.population)}')

    # emigration
    numb_index_m, numb_index_f = [], []
    for i in range(17):
        numb_em_m = int(round(migr_male[i, 9 + year - 2011] * migration[1, year - 2011] * sex_male[1, year - 2011]))
        numb_em_f = int(round(migr_female[i, 9 + year - 2011] * migration[1, year - 2011] * sex_female[1, year - 2011]))
        if i < 16:
            numb_index_m.append(numb_em_m / 5)
            numb_index_f.append(numb_em_f / 5)
        else:
            numb_index_m.append(numb_em_m / 20)
            numb_index_f.append(numb_em_f / 20)

    numb_index_m.append(0)
    numb_index_f.append(0)

    age_mi = ages_mi[:4]
    age_mi[3] += 1
    numb_index = numb_index_m[:4]
    numb_index[3] = numb_index[2]
    temp_m = [lagrange(j, age_mi, numb_index) for j in range(18)]
    temp_m = [i if i > 0 else 0 for i in temp_m]

    age_mi = ages_mi[3:]
    age_mi[0] += 1
    numb_index = numb_index_m[3:]
    k = numb_index[0] * 5
    migr = 0
    for i in range(int(numb_index[0]), int(numb_index[0] * 5)):
        migr_age = [18, 22, 27]
        migr_m = [i, numb_index[1], numb_index[2]]
        temp = [lagrange(j, migr_age, migr_m) for j in range(18, 20)]
        temp = [d if d > 0 else 0 for d in temp]
        if k > abs(numb_index[0] * 5 - sum(temp) - sum(temp_m[15:])):
            k = abs(numb_index[0] * 5 - sum(temp) - sum(temp_m[15:]))
            migr = i
    numb_index[0] = migr
    temp = [lagrange(j, age_mi, numb_index) for j in range(18, 100)]
    [temp_m.append(i) for i in temp]
    temp_m = [i if i > 0 else 0 for i in temp_m]

    age_mi = ages_mi[:4]
    age_mi[3] += 1
    numb_index = numb_index_f[:4]
    numb_index[3] = numb_index[2]
    temp_f = [lagrange(j, age_mi, numb_index) for j in range(18)]
    temp_f = [i if i > 0 else 0 for i in temp_f]

    age_mi = ages_mi[3:]
    age_mi[0] += 1
    numb_index = numb_index_f[3:]
    k = numb_index[0] * 5
    migr = 0
    for i in range(int(numb_index[0]), int(numb_index[0] * 5)):
        migr_age = [18, 22, 27]
        migr_f = [i, numb_index[1], numb_index[2]]
        temp = [lagrange(j, migr_age, migr_f) for j in range(18, 20)]
        temp = [d if d > 0 else 0 for d in temp]
        if k > abs(numb_index[0] * 5 - sum(temp) - sum(temp_f[15:])):
            k = abs(numb_index[0] * 5 - sum(temp) - sum(temp_f[15:]))
            migr = i
    numb_index[0] = migr
    temp = [lagrange(j, age_mi, numb_index) for j in range(18, 100)]
    [temp_f.append(i) for i in temp]
    temp_f = [i if i > 0 else 0 for i in temp_f]

    curr_index = []
    for i in range(1, 100):
        curr = data[data.age == i]
        curr_m = curr[curr.sex == 'M']
        curr_f = curr[curr.sex == 'F']
        if len(curr_m) < int(round(temp_m[i])):
            temp_m[i] = len(curr_m)
        real_emi = np.random.choice(np.array(curr_m.sp_id), int(round(temp_m[i])), replace=False)
        curr_index.extend(real_emi)
        if len(curr_f) < int(round(temp_f[i])):
            temp_f[i] = len(curr_f)
        real_emi = np.random.choice(np.array(curr_f.sp_id), int(round(temp_f[i])), replace=False)
        curr_index.extend(real_emi)

    print(Counter(data.loc[data.sp_id.isin(curr_index), 'factor_age']))
    data = data[~data.sp_id.isin(curr_index)]

    population.mortality_emigration(curr_index)
    print(f'len population: {len(population.population)}')
    
    emigration_sim.append(size - len(data))

    print(len(data), 'после эмиграции', size - len(data))
    size = len(data)

    print('Свободных квартир:', len(population.empty_aparts))
    print('Количество свободных мест в квартирах:', sum([population.Aparts[i].get_free() for i in population.empty_aparts]))

    print('Свободных работ:', len(population.empty_workplaces))
    print('Количество свободных рабочих мест:', sum([population.WorkPlaces[i].get_free() for i in population.empty_workplaces]))


    # immigration
    new = pd.DataFrame()
    max_id = max(data.sp_id)
    numb_index_m, numb_index_f = [], []
    for i in range(17):
        numb_im_m = int(round(migr_male[i, year - 2011] * migration[0, year - 2011] * sex_male[0, year - 2011]))
        numb_im_f = int(round(migr_female[i, year - 2011] * migration[0, year - 2011] * sex_female[0, year - 2011]))
        if i < 16:
            numb_index_m.append(numb_im_m / 5)
            numb_index_f.append(numb_im_f / 5)
        else:
            numb_index_m.append(numb_im_m / 20)
            numb_index_f.append(numb_im_f / 20)
    numb_index_m.append(0)
    numb_index_f.append(0)

    age_mi = ages_mi[:4]
    age_mi[3] += 1
    numb_index = numb_index_m[:4]
    numb_index[3] = numb_index[2]
    temp_m = [lagrange(j, age_mi, numb_index) for j in range(18)]
    temp_m = [i if i > 0 else 0 for i in temp_m]

    age_mi = ages_mi[3:]
    age_mi[0] += 1
    numb_index = numb_index_m[3:]
    k = numb_index[0] * 5
    migr = 0
    for i in range(int(numb_index[0]), int(numb_index[0] * 5)):
        migr_age = [18, 22, 27]
        migr_m = [i, numb_index[1], numb_index[2]]
        temp = [lagrange(j, migr_age, migr_m) for j in range(18, 20)]
        temp = [d if d > 0 else 0 for d in temp]
        if k > abs(numb_index[0] * 5 - sum(temp) - sum(temp_m[15:])):
            k = abs(numb_index[0] * 5 - sum(temp) - sum(temp_m[15:]))
            migr = i
    numb_index[0] = migr
    temp = [lagrange(j, age_mi, numb_index) for j in range(18, 100)]
    [temp_m.append(i) for i in temp]
    temp_m = [i if i > 0 else 0 for i in temp_m]

    age_mi = ages_mi[:4]
    age_mi[3] += 1
    numb_index = numb_index_f[:4]
    numb_index[3] = numb_index[2]
    temp_f = [lagrange(j, age_mi, numb_index) for j in range(18)]
    temp_f = [i if i > 0 else 0 for i in temp_f]

    age_mi = ages_mi[3:]
    age_mi[0] += 1
    numb_index = numb_index_f[3:]
    k = numb_index[0] * 5
    migr = 0
    for i in range(int(numb_index[0]), int(numb_index[0] * 5)):
        migr_age = [18, 22, 27]
        migr_f = [i, numb_index[1], numb_index[2]]
        temp = [lagrange(j, migr_age, migr_f) for j in range(18, 20)]
        temp = [d if d > 0 else 0 for d in temp]
        if k > abs(numb_index[0] * 5 - sum(temp) - sum(temp_f[15:])):
            k = abs(numb_index[0] * 5 - sum(temp) - sum(temp_f[15:]))
            migr = i
    numb_index[0] = migr
    temp = [lagrange(j, age_mi, numb_index) for j in range(18, 100)]
    [temp_f.append(i) for i in temp]
    temp_f = [i if i > 0 else 0 for i in temp_f]

    for i in range(1, 100):
        if i < 80:
            factor = int(i / 5)
        else:
            factor = 16
        age_list = [i] * int(round(temp_m[i]))
        sex_list = ['M'] * int(round(temp_m[i]))
        id_list = list(range(max_id + 1, max_id + int(round(temp_m[i])) + 1))
        max_id += int(round(temp_m[i]))

        new_frame = pd.DataFrame({'sp_id': id_list,
                                  'age': age_list,
                                  'sex': sex_list,
                                 }
                                 
                                )
        new = new.append(new_frame, ignore_index=True)

        age_list = [i] * int(round(temp_f[i]))
        sex_list = ['F'] * int(round(temp_f[i]))
        id_list = list(range(max_id + 1, max_id + int(round(temp_f[i])) + 1))
        max_id += int(round(temp_f[i]))
        
        new_frame = pd.DataFrame({'sp_id': id_list,
                                  'age': age_list,
                                  'sex': sex_list,
                                 }
                                )
        new = new.append(new_frame, ignore_index=True)

    new = new.sample(frac=1).reset_index(drop=True)

    data = data.append(new, ignore_index=True)

    population.immigration(new) # !!! mode = ['closest', 'random']
    new['factor_age'] = new['age'].apply(factor_age)
    print(Counter(new.factor_age))
    
    imigration_sim.append(len(data) - size)

    print(len(data), 'после имиграции', len(data) - size)
    print(f'len population: {len(population.population)}')
    size = len(data)

    hundreds = data[data['age']>=100].sp_id.tolist()
    data = data[data.age < 100]
    print(len(data), 'после "столетия"', size - len(data))
    size = len(data)

    population.mortality_emigration(hundreds)
    print(f'len population: {len(population.population)}')

    output = population.create_dataframe()

    output.to_csv(os.path.join(population_folder, 'people_{}.txt'.format(year)), sep='\t')

    with open(os.path.join(init_pop_folder, f'population_{year}.pickle'), 'wb') as f:
        pickle.dump(population, f)


    end_year = time.time()
    print(f'Время расчета {year} года: {end_year-start_year} сек')


end = time.time()
print('Время выполнения:', end-start)




    

