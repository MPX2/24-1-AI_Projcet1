import pandas as pd
import numpy as np
from random import shuffle
import random
import time
import math

start_time = time.time()

# 경로 저장 class
class route:  
    def __init__(self): 
        node = Node(0,0,-3)
        self.city_route = []            # 초기 경로 리스트에 0,0 초기화
        self.city_route.append(node)    # 경로에서 첫번째 노드 즉 원점은 인덱스가 -1 이후 인덱스는 실제 데이터가 저장된 인덱스로 함.
    
    def __getitem__(self, index):
        return self.city_route[index]

    def insert(self, node):             # 노드를 리스트에 추가
        self.city_route.append(node)

    def get_terminal(self):             # 마지막 노드 반환
        return self.city_route[-1]

# 도시 하나 = Node (도시의 x y 좌표, index 번호)
class Node:
    def __init__(self, x, y, index, children=None):
        self.x = x
        self.y = y
        self.uclid = math.sqrt((self.x)**2 + (self.y)**2)
        self.index = index
        if children is None:
            self.children = []
        else:
            self.children = children
    
    def get_heuristic(self):                    # 휴리스틱
        heuristic_value = 1/ self.manhattan_distance()
        return heuristic_value

    def uclid_distance(self, node):             # 유클리드 거리
        return math.sqrt(abs(self.x - node.x)**2 + abs(self.y - node.y)**2)
    def manhattan_distance(self):               # 맨하튼 거리
        return abs(self.x) + abs(self.y)
    def get_index(self):                        # index
        return self.index
    def get_x(self):                            # x값
        return self.x
    def get_y(self):                            # y값 
        return self.y
    
#  A* 알고리즘
class A_star:
    base = route()
    route_list = []     # 한 세대 경로 집합
    node_data = []

    def __init__(self, base, route_count): 
        self.route_count = route_count
        self.base = base
        for i in range(0, len(base.city_route)):
            self.node_data.append(base.city_route[i])       
        for _ in range(0,route_count):
            self.route_list.append(route())
    
    def __getitem__(self, index):
        return self.route_list[index]

여기 수정 필요
    def A_star(self):       # A*로 경로 결정
        for i in range(0,self.route_count):
            copy_array = self.node_data[:]
            current_Node = copy_array[i] 
            current_index = i
            while len(copy_array) > 1:
                max = 0
                next_index = -1

                if current_index-50<0 and current_index+50>len(copy_array)-1:
                    for k in range(0, len(copy_array)):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산 
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.route_list[i].insert(copy_array[current_index])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1
                    
                elif current_index-50<0:
                    for k in range(0,current_index):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산 
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.route_list[i].insert(copy_array[current_index])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1

                elif current_index+50>len(copy_array)-1:
                    for k in range(current_index-20,len(copy_array)):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산 
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.route_list[i].insert(copy_array[current_index])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1
        
                else:
                    for k in range(current_index-50,current_index+50):
                        if k == current_index:
                            continue
                        p_p_list = current_Node.uclid_distance(copy_array[k]) #실제 다음 후보노드까지의 거리계산 
                        heuristic_value = copy_array[k].get_heuristic()#휴리스틱계산
                        fn_value = p_p_list + heuristic_value #f(N)계산
                        if fn_value > max:
                            next_index = k
                            max = fn_value
                    self.route_list[i].insert(copy_array[current_index])
                    current_Node = copy_array[next_index]
                    copy_array.pop(current_index)
                    current_index = next_index
                    if current_index != 0:
                        current_index -= 1

# CSV 파일에서 좌표 데이터를 가져옴
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    coordinates = df.values
    return coordinates


################ 초기 해 생성 ################
def initial(base, population_size):
    return initial_population(base, population_size)
    # return astar_population(base, population_size)

def sort(coordinates):          
    # 기준 route, 거리로 sort하여 순서대로 저장
    data_list = []
    sorted_route = route()
    for i in range(0, len(coordinates)):
        data_list.append([coordinates.iat[i,0], coordinates.iat[i,1]])
    data_list.sort(key=lambda x: abs(x[1])+abs(x[0]))

    for city_index in range(0,len(data_list)):
        new_node = Node(data_list[city_index][0],data_list[city_index][1],city_index)
        sorted_route.insert(new_node)
    
    return sorted_route

def initial_population(base, population_size):     
    # 노드를 랜덤한 순서로 넣어 초기값
    population = []
    for _ in range(population_size):
        new_route = route()
        path = list(base[1:])
        shuffle(path)
        path.insert(0, base[0])
        for city in path:
            new_route.insert(city)
        population.append(new_route)
    return population

def astar_population(base, population_size):
     에이스타 시작해 추가

################ Evaluation Function ################

# 좌표 간 거리 계산
def distance(city1, city2):
    return math.sqrt((city1.get_x() - city2.get_x())**2 + (city1.get_y() - city2.get_y())**2)

def fitness(population): 
    # 적합도 함수
    fitness_values = []
    for individual in population:
        total_distance = 0
        index_gap = 0
        for i in range(len(individual.city_route) - 1):
            node1 = individual[i]
            node2 = individual[i+1]
            total_distance += distance(node1, node2)

            if abs(i - node1.get_index()) < 10:
                index_gap += 20
            elif abs(i - node1.get_index()) < 20:
                index_gap += 5

        node_last = individual.get_terminal()
        node_first = individual[0]
        total_distance += distance(node_last, node_first)
        fitness_values.append(1 / total_distance + 1 / index_gap)

    return fitness_values


# def distance_calculate(population, coordinates): # 적합도 함수 -> 총 거리 합의 역수
#    distance_values = []
#    for individual in population:
#        total_distance = 0
#        for i in range(len(individual) - 1):
#            city1 = coordinates[individual[i]]
#            city2 = coordinates[individual[i + 1]]
#            total_distance += distance(city1, city2)
#        total_distance += distance(coordinates[individual[-1]], coordinates[individual[0]])
#        distance_values.append(total_distance)
#    return sum(distance_values) / len(distance_values)

################ Selection ################
def selection(population, fitness_values):
    return tournament_selection(population, fitness_values)


def tournament_selection(population, fitness_values):
    tournament_size = 5
    selection_pressure = 0.9

    tournament = random.sample(range(len(population)), tournament_size)  # 토너먼트 크기만큼 무작위로 개체를 선택
    tournament_fitness = [fitness_values[i] for i in tournament]
    if random.random() < selection_pressure:
        tournament_winner = tournament[np.argmax(tournament_fitness)]  # 토너먼트에서 가장 적합도가 높은 개체의 인덱스 선택
    else:
        tournament_winner = tournament[np.argmin(tournament_fitness)]  # 토너먼트에서 가장 적합도가 낮은 개체의 인덱스 선택
    return population[tournament_winner]


################ Crossover ################
def crossover(parent1, parent2, crossover_rate):
     return singlepoint_crossover(parent1, parent2, crossover_rate)
    # return uniform_crossover(parent1, parent2, crossover_rate)
    # return cycle_crossover(parent1, parent2, crossover_rate)
    # return er_crossover(parent1, parent2, crossover_rate)


def singlepoint_crossover(parent1, parent2, crossover_rate):

    def dup_indices(arr1, arr2):
        duplicate_indices = []
        
        # 배열2의 각 값에 대해 반복
        for index in range(len(arr2)):
            value = arr2[index]
            
            if value in arr1:
                duplicate_indices.append(index)
        
        return duplicate_indices
    
    def find_missingValues(array):
        # 집합으로 변환
        unique_values = set(array)

        complete_set = set(range(998))
        missing_values = sorted(complete_set - unique_values)
        
        return missing_values
    
    def singlepoint_func(parent1, parent2):
        crossover_point = random.randint(10, len(parent1.city_route) - 10)
        temparr = parent2.city_route[crossover_point:]
        duplicate_indices = dup_indices(parent1.city_route[:crossover_point], temparr)
        incomplete_arr = parent1.city_route[:crossover_point] + parent2.city_route[crossover_point:]
        missing_values = find_missingValues(incomplete_arr)
        if len(duplicate_indices) == len(missing_values):
            for i in range(len(duplicate_indices)):
                temparr[duplicate_indices[i]] = missing_values[i]
        else:
            print("len(duplicate_indices) =/= len(missing_values)!")
            exit(1)
        return parent1[:crossover_point] + temparr

    if random.random() < crossover_rate:
        child1 = singlepoint_func(parent1, parent2)
        child2 = singlepoint_func(parent2, parent1)
        return child1, child2
    else:
        return parent1, parent2


def uniform_crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    else:
        return parent1, parent2
    
def cycle_crossover(parent1,parent2,crossover_rate):
    if random.random() < crossover_rate:
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent1)
        cycle_num = 0
        visited = [False] * len(parent1)

        while False in visited:
            if cycle_num % 2 == 0:
                dest1 = child1
                dest2 = child2
            else:  
                dest1 = child2
                dest2 = child1

            for start in range(len(parent1)):
                if not visited[start]:
                    break
            current = start
            while True:
                dest1[current] = parent1[current]
                dest2[current] = parent2[current]
                visited[current] = True
                current = parent1.index(parent2[current])
                if current == start:
                    break
            cycle_num += 1
        return child1, child2
    
    else:
        return parent1, parent2
    
def er_crossover(parent1,parent2,crossover_rate):
    if random.random() < crossover_rate:
        num_cities = len(parent1)
        child1 = [None] * num_cities
        child2 = [None] * num_cities
        edges1 = {i: set() for i in range(num_cities+1)}
        edges2 = {i: set() for i in range(num_cities+1)}
        def add_edge(parent): #인접도시 목록 만들기
          for i in range(num_cities):
            left=parent[(i-1)%num_cities]
            right=parent[(i+1)%num_cities]
            edges1[parent[i]].update([left,right])
            edges2[parent[i]].update([left,right])
        add_edge(parent1)
        add_edge(parent2)
        
        current = parent1[0]
        child1 = [current]
        while len(child1) < len(parent1): 
            for edges in edges1.values():
                edges.discard(current) #방문한 도시는 인접도시 목록에서 삭제
            if edges1[current]:
                next_city = min(edges1[current], key=lambda x: len(edges1[x])) #현재 노드에서 인접 도시가 있다면 그중 남은 인접 도시가 가장 적은 지역 우선 방문
            else:
                remaining = set(parent1) - set(child1)
                next_city = random.choice(list(remaining)) #현재 노드에서 인접 도시가 없다면 전체에서 랜덤 방문
            child1.append(next_city)
            current = next_city

        current = parent2[0]
        child2 = [current]
        while len(child2) < len(parent2):
            for edges in edges2.values():
                edges.discard(current)
            if edges2[current]:
                next_city = min(edges2[current], key=lambda x: len(edges2[x]))
            else:
                remaining = set(parent2) - set(child2)
                next_city = random.choice(list(remaining))
            child2.append(next_city)
            current = next_city

        return child1,child2
    else:
        return parent1,parent2


def a_star_crossover(route1, route2,num): #임의의 난수 뽑아서 인덱스 순으로 정렬 route1,route2는 경로2개이고 num은 997개의 점중에 몇개를 바꿀건지정하는 정수
    sort_list = []
    zero_node = Node(0,0,-1)
    index_num = random.randint(1,997-num)
    for i in range(index_num, index_num+num):
        sort_list.append(route1.city_route[i])
    
    sort_list.sort(key = lambda x :x.get_index()+2*x.uclid_distance(zero_node))

    for i in range(0,num):
        route1.city_route[i] = sort_list[i]

    sort_list2 = []
    index_num = random.randint(1,997-num)

    for i in range(index_num, index_num+num):
        sort_list2.append(route2.city_route[i])
    
    sort_list2.sort(key = lambda x :x.get_index()+2*x.uclid_distance(zero_node))

    for i in range(0,num):
        route2.city_route[i] = sort_list2[i]
    
    

################ Mutation ################
def mutation(individual, mutation_rate):
    return notequal_mutation(individual, mutation_rate)

def notequal_mutation(individual, cur_generation):
    # # 1개의 유전자 선택 후 서로 교환
    # if random.random() < 0.1 / math.log10(10 + cur_generation):
    #     idx1, idx2 = random.sample(range(1, len(individual)), 2)
    #     individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    # return individual

    # 10개의 유전자를 선택해서 서로 교환
    if random.random() < 0.1 / math.log10(10 + cur_generation):
        indices = random.sample(range(1, len(individual)), 10)
        for i in range(0, len(indices), 2):
            idx1, idx2 = indices[i], indices[i+1]
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    
    return individual
   

################ GA ################
def genetic_algorithm(coordinates, base, population_size, generations, crossover_rate, mutation_rate):
    population = initial(base, population_size)
    best_distance = float('inf')
    best_path = None
    # end_counter = 0
    # genAvgDistance = []

    for cur_generation in range(generations):
        fitness_values = fitness(population)
        new_population = []
    #    genAvgDistance.append(distance_calculate(population, coordinates))
    #   if cur_generation > 500:
    #    if abs(genAvgDistance[cur_generation - 1] - genAvgDistance[cur_generation]) < 0.15:
    #        end_counter += 1
    #    else:
    #        end_counter = 0
                
    #    if end_counter > 10:
    #        print("Program ended at generation %d" % (cur_generation + 1))                
    #        return best_path
        
        for _ in range(population_size // 2):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = notequal_mutation(child1, cur_generation)
            child2 = notequal_mutation(child2, cur_generation) # 교차 과정으로 2개의 자식이 생성
            new_population.extend([child1, child2])

        population = new_population
        min_distance = 1 / max(fitness_values)  # 최소 거리는 최대 적합도의 역수
        if min_distance < best_distance:
            best_distance = min_distance
            best_path = population[np.argmax(fitness_values)]
        print("current generation : %d" % (cur_generation + 1))

    print("Program ended at generation %d" % (cur_generation + 1))
    return best_path

# 데이터 로드
coordinates = load_data("2024_AI_TSP.csv")
coordinates_df = pd.DataFrame(coordinates)
base_route = sort(coordinates_df)

# 설정
population_size = 50 # 개체군의 크기(초기에 생성되는 경로의 개수)
generations = 2000 # 수행할 반복 횟수
crossover_rate = 0.8
mutation_rate = 0.05 # 다양성 증가, local optimal 방지

# 유전 알고리즘 실행
best_path = genetic_algorithm(coordinates, base_route, population_size, generations, crossover_rate, mutation_rate)

# 최적 경로 저장
indexes = []
for node in best_path:
    x = node.get_x()
    y = node.get_y()
    for i in range(len(coordinates)):
        if coordinates.iat[i,0] == x and coordinates.iat[i,1] == y:
            indexes.append(i)

best_path_df = pd.DataFrame(indexes)
#best_path_df = pd.DataFrame(best_path)
best_path_df.to_csv("best_path.csv", index=False, header=False)
print("\n")
print("best_path.csv' 저장 성공")
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time
print("실행시간 :", execution_time, "초")

# 실제 거리 평가
import TSP_eval