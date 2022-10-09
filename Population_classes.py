import pandas as pd
import numpy as np
import json
import os
import random

def read_building(path):
    b = pd.read_csv(path, 
                    sep='\t', 
                    dtype={'sp_id': np.int32},
                    usecols=['sp_id', 'latitude', 'longitude'])
    b['sp_id'] = b['sp_id'].astype(str)
    b['latitude'] = b['latitude'].round(6)
    b['longitude'] = b['longitude'].round(6)
    return b

def employed_status(age, sex):
    status = 0 # для детей <7 пенсионеров >65(M) или >60(F)
    if (sex == 'F') and (age > 17) and (age < 60): # для работающих женщин
        status = 2
    elif (sex == 'M') and (age > 17) and (age < 65): # для работающих мужчин
        status = 2
    elif (age >= 7) and (age <= 17):
        status = 1 # для учащихся
    return status

def set_factor(age):
    if age < 80:
        factor = age // 5
    else:
        factor = 16
    return factor


class Person:
    def __init__(self, person_id, age, sex, apart_id=None, school_id=None, work_id=None):
        self.person_id = person_id
        self.age = age
        self.sex = sex
        self.status = employed_status(self.age, self.sex)
        self.factor = set_factor(self.age)
        self.apart_id = apart_id
        self.school_id = school_id
        self.work_id = work_id
        self.hh_id = None


class Building:
    def __init__(self, bid, coords, capacity=None):
        self.bid = bid
        self.coords = coords
        self.capacity = capacity
        self.current_capacity = 0
        
class School(Building):
    def __init__(self, bid, coords, capacity=1200):
        Building.__init__(self, bid, coords, capacity)
#     def __init__(self, capacity=1200):
#         self.capacity = capacity

class HouseHold(Building):
    pass
#    def __init__(self, bid, coords, capacity=None):
#        Building.__init__(self, bid, coords)
        #self.geoid_to_workid = geoid_to_workid

    # def get_closest_schools(self, hh_schools_5km, hh_schools_5_15km):
    #     if hh_schools_5km.get(self.bid):
    #         self.closest_schools = hh_schools_5km[self.bid]
    #     else:
    #         self.closest_schools = hh_schools_5_15km[self.bid] # перезаписать json отсортированными по расст школами(done)
            
    # def get_closest_geoids(self, hh_works_15km_sorted): #, hh_works_5_15km):
    #     if hh_works_15km_sorted.get(self.bid):
    #         self.closest_geoids = hh_works_15km_sorted[self.bid] # bid == hh id
            # cw = []
            # for geoid in self.closest_geoids:
            #     cw += geoid_to_workid[geoid]
            # self.closest_works = cw
class Apart(Building):
    def set_capacity(self):
        if self.current_capacity > 0:
            self.capacity = self.current_capacity
            
        else:
            self.capacity = np.random.poisson(3)+3
            # np.random.choice(np.arange(1, 11), 
            #                                 p=np.array([2.08643461e-01, 3.25768417e-01, 2.62162867e-01, 1.28061226e-01,
            #                                             5.33540158e-02, 1.84435976e-02, 2.82374134e-03, 3.30252904e-04,
            #                                             3.27092589e-04, 8.53285016e-05]))
            #self.capacity = np.random.randint(1, 7+1) # взять другое распределение на основе уже сущ заселения
    def get_free(self):
        return self.capacity - self.current_capacity

class WorkPlace(Building):
    def set_capacity(self):
        if self.current_capacity > 0:
            self.capacity = self.current_capacity
            
        else:
            self.capacity = np.random.poisson(7) + 1

    def get_free(self):
        return self.capacity - self.current_capacity

class Population():
    def __init__(self, Schools, WorkPlaces, Aparts, HouseHolds, apart_to_hh, 
                    empty_aparts, empty_workplaces, hh_schools_5km, hh_schools_5_15km, 
                    hh_works_15km_sorted, geoid_to_workid, mode='random'
                    ):
        self.population = {}
        self.Schools = Schools
        self.WorkPlaces = WorkPlaces
        self.Aparts = Aparts
        self.HouseHolds = HouseHolds
        self.apart_to_hh = apart_to_hh
        self.empty_aparts = empty_aparts
        self.empty_workplaces = empty_workplaces
        self.hh_schools_5km = hh_schools_5km
        self.hh_schools_5_15km = hh_schools_5_15km
        self.hh_works_15km_sorted = hh_works_15km_sorted
        self.geoid_to_workid = geoid_to_workid
        self.mode = mode
    
    def add_person(self, person):
        self.population[person.person_id] = person
        
    def remove_person(self, person_id):#, Schools, WorkPlaces, Aparts):
        person = self.population.pop(person_id)     

        self.Aparts[person.apart_id].current_capacity -= 1
        if self.Aparts[person.apart_id].current_capacity < self.Aparts[person.apart_id].capacity:
            self.empty_aparts.add(person.apart_id)

        if person.school_id:
            self.Schools[person.school_id].current_capacity -= 1

        elif person.work_id:
            self.WorkPlaces[person.work_id].current_capacity -= 1
            self.empty_workplaces.add(person.work_id) # add workplace to available workplaces

    def create_dataframe(self):
        df = pd.DataFrame()
        len_pop = len(self.population.values())
        sp_id = []
        sp_hh_id = []
        age = []
        sex = []
        factor_age = []
        work_id = []
        for i, person in enumerate(self.population.values()):
            # row = pd.DataFrame({'sp_id': [person.person_id], 
            # 'sp_hh_id': [person.apart_id], 
            # 'age': [person.age], 
            # 'sex': [person.sex], 
            # 'factor_age': [person.factor]
            # }
            # )
            # df = df.append(row, ignore_index=True)

            sp_id.append(person.person_id) 
            sp_hh_id.append(person.apart_id) 
            age.append(person.age) 
            sex.append(person.sex) 
            factor_age.append(person.factor)
            if person.school_id:
                work_id.append(person.school_id)
            elif person.work_id:
                work_id.append(person.work_id)
            else:
                work_id.append('X')

            if i%100==0:
                print(f'{i}/{len_pop}', end='\r')
        print()
        df = pd.DataFrame({'sp_id': sp_id, 
            'sp_hh_id': sp_hh_id, 
            'age': age, 
            'sex': sex, 
            'factor_age': factor_age,
            'work_id': work_id
            }, 
            )
        df['sp_id'] = df['sp_id'].astype(int)
        df['sp_hh_id'] = df['sp_hh_id'].astype(str)
        df['age'] = df['age'].astype(int)
        df['sex'] = df['sex'].astype(str)
        df['factor_age'] = df['factor_age'].astype(int)
        df['work_id'] = df['work_id'].astype(str)
        return df

    def mortality_emigration(self, mortal_list):
        for person_id in mortal_list:
            self.remove_person(person_id)
        
    def fertility(self, birth_df):
        birth_df_dict = birth_df.to_dict('records')
        for i, p in enumerate(birth_df_dict):
            person = self.create_newborn(p)
            self.add_person(person)
            self.Aparts[person.apart_id].current_capacity += 1
        
    def immigration(self, immigration_df):
        pop_len=len(immigration_df)
        empap = list(self.empty_aparts)
        random.shuffle(empap)
        empap = empap[:pop_len]
        #apart_id = random.choice(empap)
        immigration_df_dict = immigration_df.to_dict('records')
        for i, (p, apart_id) in enumerate(zip(immigration_df_dict, empap)):
            person = self.create_immigrant(p, apart_id)
            #print(person.__dict__)
            self.add_person(person)
            self.Aparts[person.apart_id].current_capacity += 1
            if self.Aparts[person.apart_id].current_capacity == self.Aparts[person.apart_id].capacity:
                self.empty_aparts.discard(person.apart_id)
            if person.work_id:
                self.WorkPlaces[person.work_id].current_capacity += 1
                if self.WorkPlaces[person.work_id].current_capacity == self.WorkPlaces[person.work_id].capacity:
                    self.empty_workplaces.discard(person.work_id)
            elif person.school_id:
                self.Schools[person.school_id].current_capacity += 1


            if i % 100 ==0 :
                print(f'{i}/{pop_len}', end='\r')

    def increase_year(self, person):
        status = person.status
        person.age += 1
        person.factor = set_factor(person.age)
        person.status =  employed_status(person.age, person.sex) 
        if status == 0 and person.status == 1:
            try:
                school_id = self.set_school(person.hh_id)
                person.school_id = school_id
                self.Schools[school_id].current_capacity += 1
            except:
                print(person.__dict__)
        elif status == 1 and person.status == 2:
            self.Schools[person.school_id].current_capacity -= 1
            person.school_id = None
            work_id = self.find_closest_works(person.hh_id)
            if work_id:
                person.work_id = work_id
                self.WorkPlaces[work_id].current_capacity += 1
                if self.WorkPlaces[work_id].current_capacity == self.WorkPlaces[work_id].capacity:
                    self.empty_workplaces.discard(work_id)
            #self.set_workplace(WorkPlaces, HouseHolds, initial_pop=False)
            #print(f'set school {self.school_id} and work {self.work_id}')
        elif status == 2 and person.status == 0:
            if person.work_id:
                self.WorkPlaces[person.work_id].current_capacity -= 1 # Добавить работу в доступные
                self.empty_workplaces.add(person.work_id)
                person.work_id = None

    def update_year(self):#, Schools, WorkPlaces, HouseHolds):
        print('Update new year...')
        pop_len = len(self.population)
        for i, person in enumerate(self.population.values()):
            self.increase_year(person)
            #person.increase_year(self.Schools, self.WorkPlaces, self.HouseHolds, initial_pop=False)
            # !!! обновить работы, апарты и школы
            # if person.work_id:
            #     if self.WorkPlaces[person.work_id].current_capacity == self.WorkPlaces[person.work_id].capacity:
            #         self.empty_workplaces.discard(person.work_id)
            if i % 100 == 0:
                print(f'{i}/{pop_len}', end='\r')
        print()



    def create_population_from_csv(self, data):#, Schools, WorkPlaces, Apartments, HouseHolds, apart_to_hh):
        print('Creating population...')
        data_employed = data[data['work_id']!='X']
        data_emp_dict = data_employed.to_dict('records')
        for i, p in enumerate(data_emp_dict):
            person = self.create_person(p) #, initial_pop=True)
            self.add_person(person)

            self.Aparts[person.apart_id].current_capacity += 1
            x = self.Aparts[person.apart_id].current_capacity
            self.Aparts[person.apart_id].capacity = x + 1 if x>=3 else 3
            #if person.apart_id in self.empty_aparts:
            self.empty_aparts.discard(person.apart_id)

            if person.school_id:
                self.Schools[person.school_id].current_capacity +=1
                
            elif person.work_id:
                self.WorkPlaces[person.work_id].current_capacity +=1
                self.WorkPlaces[person.work_id].capacity = self.WorkPlaces[person.work_id].current_capacity
                #if person.work_id in self.empty_workplaces:
                self.empty_workplaces.discard(person.work_id)

            if i%100 == 0:
                print(f'{i}/{len(data_employed)}', end='\r')
        print()

        print('Set capacity WP:', len(self.empty_workplaces))      
        for workplace_id in self.empty_workplaces:
            self.WorkPlaces[workplace_id].set_capacity()

        data_unemployed = data[data['work_id']=='X']
        data_unemp_dict = data_unemployed.to_dict('records')
        for i, p in enumerate(data_unemp_dict):
            person = self.create_person(p) #, initial_pop=True)
            self.add_person(person)

            self.Aparts[person.apart_id].current_capacity += 1
            x = self.Aparts[person.apart_id].current_capacity
            self.Aparts[person.apart_id].capacity = x + 1 if x>=3 else 3
            #if person.apart_id in self.empty_aparts:
            self.empty_aparts.discard(person.apart_id)

            if person.school_id:
                self.Schools[person.school_id].current_capacity +=1
                
            elif person.work_id:
                self.WorkPlaces[person.work_id].current_capacity +=1
                if self.WorkPlaces[person.work_id].current_capacity == self.WorkPlaces[person.work_id].capacity:
                    self.empty_workplaces.discard(person.work_id)

            if i%100 == 0:
                print(f'{i}/{len(data_unemployed)}', end='\r')
        print()

        print('Set empty aparts capacity:', len(self.empty_aparts))
        for a in self.empty_aparts:
            self.Aparts[a].set_capacity()

    def set_household(self, apart_id):
        return self.apart_to_hh[apart_id]

    def find_closest_schools(self, hh_id):
        closest_schools = []
        if self.hh_schools_5km.get(hh_id):
            closest_schools = self.hh_schools_5km[hh_id]
        else:
            closest_schools = self.hh_schools_5_15km[hh_id]
        return closest_schools

    def set_school(self, hh_id):
        closest_schools = self.find_closest_schools(hh_id)
        if self.mode=='random':
            random.shuffle(closest_schools)

        for school_id in closest_schools:
            if self.Schools[school_id].current_capacity < 1200:
                return school_id
        return closest_schools[0]
                # self.school_id = school_id
                # Schools[school_id].current_capacity += 1
                # break

    def find_closest_works(self, hh_id, initial_pop=False):
        if self.hh_works_15km_sorted.get(hh_id):
            closest_geoids = self.hh_works_15km_sorted[hh_id]
            if self.mode=='random':
                random.shuffle(closest_geoids)
            for geoid in closest_geoids:
                workids_from_geoids = self.geoid_to_workid[geoid]
                workids_from_geoids = set(workids_from_geoids).intersection(self.empty_workplaces)
                if self.mode=='random':
                    random.shuffle(workids_from_geoids)
                for workid in workids_from_geoids:
                    if self.WorkPlaces.get(workid):
                        if initial_pop:
                            return workid
                        #break
                        else:
                            if self.WorkPlaces[workid].current_capacity < self.WorkPlaces[workid].capacity:
                                return workid
                            #WorkPlaces[workplace_id].current_capacity += 1 # ДОБАВИТЬ ПРОВЕРКУ ОСТАЛАСЬ ЛИ РАБОЧЕЕ МЕСТО ДОСТУПНЫМ!!!
                            #break
        return None
    
    def create_person(self, person_data):
        person_id = person_data['sp_id']
        age = person_data['age']
        sex = person_data['sex']
        apart_id = person_data.get('sp_hh_id')
        person = Person(person_id, age, sex, apart_id)
        
        # if not apart_id:
        #     empap = list(self.empty_aparts)
        #     apart_id = random.choice(empap)
        #     #print(apart_id)
        
#        person.apart_id = apart_id
        hh_id = self.set_household(person.apart_id)
        person.hh_id = hh_id
        work_id = person_data.get('work_id') if person_data.get('work_id') != 'X' else None
        if work_id:
            if person.status == 1:
                person.school_id = work_id
            elif person.status == 2:
                person.work_id = work_id
        else:
            if person.status == 1:
                school_id = self.set_school(hh_id)
                person.school_id = school_id
            elif person.status == 2:
                work_id = self.find_closest_works(hh_id)
                person.work_id = work_id
            #person.define_employment(self.Schools, self.WorkPlaces, self.HouseHolds, initial_pop)
        
        return person

    def create_newborn(self, person_data):
        person_id = person_data['sp_id']
        age = person_data['age']
        sex = person_data['sex']
        apart_id = person_data.get('sp_hh_id')
        person = Person(person_id, age, sex, apart_id)
        person.hh_id = self.set_household(person.apart_id)
        return person

    def create_immigrant(self, person_data, apart_id):
        person_id = person_data['sp_id']
        age = person_data['age']
        sex = person_data['sex']

        person = Person(person_id, age, sex, apart_id)
        hh_id = self.set_household(person.apart_id)
        person.hh_id = hh_id
        if person.status == 1:
            school_id = self.set_school(hh_id)
            person.school_id = school_id
        elif person.status == 2:
            work_id = self.find_closest_works(hh_id)
            person.work_id = work_id
        return person



# def create_school(school_data):
#     bid = school_data['sp_id']
#     coords = school_data['latitude'], school_data['longitude']
#     return School(bid, coords)

# def create_household(hh_data):#, geoid_to_workid):
#     bid = hh_data['sp_id']
#     coords = hh_data['latitude'], hh_data['longitude']
#     return HouseHold(bid, coords)#, geoid_to_workid)

# def create_workplace(workplace_data):
#     bid = workplace_data['sp_id']
#     coords = workplace_data['latitude'], workplace_data['longitude']
#     return WorkPlace(bid, coords)

# def create_apart(aparts_data):
#     bid = aparts_data['sp_id']
#     coords = aparts_data['latitude'], aparts_data['longitude']
#     #capacity = 1200
#     return Apart(bid, coords)

def create_object(data, return_class=Building):
    bid = data['sp_id']
    coords = data['latitude'], data['longitude']
    #capacity = 1200
    return return_class(bid, coords)

def create_objects_dict(data, return_class=Building):
    N = len(data)
    n = len(str(N)) - 1
    thresh = 10 ** n
    print(f'Create {return_class.__name__}s...')
    objects = {}
    data_dict = data.to_dict('records')
    for i, o in enumerate(data_dict):
        obj = create_object(o, return_class)
        objects[obj.bid] = obj
        if i%thresh == 0:
            print(f'{i}/{N}', end='\r')
    print()
    return objects
