import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Particle:
    def __init__(self, path):
        self.path = path
        self.cost = np.inf

        self.velocity = np.zeros(self.path.shape)

        # best memory of each particle is initialized as itself 
        self.pbest = self.path
        self.pbest_cost = self.cost

    def adjust_path(self, map):
        self.path = np.clip(self.path, [0, 0], [map.shape[1] - 1, map.shape[0] - 1])


    # def evaluate_cost(self, map, start, goal):
    #     # print(start.shape, self.path.shape, goal.shape)
    #     start = np.array(start).reshape(1, -1)  # Reshape to (1, 2)
    #     goal = np.array(goal).reshape(1, -1)    # Reshape to (1, 2)
    #     complete_path = np.concatenate((start, self.path, goal), axis=0)
    #     cost = 0
    #     for i in range(len(complete_path)-1):
    #         cost += self.distance(complete_path[i], complete_path[i+1])
    #         if self.is_in_obstacle(complete_path[i], complete_path[i+1], map):
    #             cost = np.inf
    #             break
    #     self.cost = cost

    def evaluate_cost(self, map, start, goal):
        complete_path = np.concatenate((start.reshape(1, -1), self.path, goal.reshape(1, -1)), axis=0)
        self.cost = np.sum(np.linalg.norm(np.diff(complete_path, axis=0), axis=1))
        for i in range(len(complete_path)-1):
            if self.is_in_obstacle(complete_path[i], complete_path[i+1], map): # Obstacle detected
                self.cost += 1e6  # Add a large penalty
                break

    def update_pbest(self):
        if self.cost < self.pbest_cost:
            self.pbest = self.path
            self.pbest_cost = self.cost

    def is_in_obstacle(self, start, end, map):
        unit_vec, distance = self.vector_between_2point(start, end)
        test_point = np.array([0, 0])
        for i in range(int(distance)):
            # print(unit_vec.shape, distance.shape, start.shape, end.shape)
            test_point[0] = start[0] + i*unit_vec[0]
            test_point[1] = start[1] + i*unit_vec[1]
            if map[math.floor(test_point[1])][math.floor(test_point[0])] == 1:
                return True   
        return False
    
    def distance(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def vector_between_2point(self, start, end):
        v = np.array([end[0]-start[0], end[1]-start[1]])
        distance = np.linalg.norm(v)
        unit_vec = v/distance
        return (unit_vec, distance)

class PSO:
    def __init__(self, map, start, goal, num_iterations, num_particles, num_waypoints, w, c1, c2):

        self.map = map
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints

        # PSO parameters
        self.num_iterations = num_iterations
        self.num_particles = num_particles
        
        self.gbest = None
        self.gbest_cost = np.inf

        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # solutions
        self.particles = []

    def run(self):

        # initialization random particles
        is_init = False
        while not is_init:
            print("Initializing...")
            for i in range(self.num_particles):
                random_path = self.create_random_path()
                particle = Particle(random_path)
                particle.evaluate_cost(self.map, self.start, self.goal)
                particle.update_pbest()

                if particle.cost < self.gbest_cost:
                    # print('found a feasible path', particle.path)
                    # print('with the cost: ', particle.cost)
                    self.gbest = particle.path
                    self.gbest_cost = particle.cost
                    is_init = True

                self.particles.insert(0, particle)
                self.particles = self.particles[:self.num_particles]

        # searching
        for i in range (self.num_iterations):
            for particle in self.particles:
                # update vi, pi
                particle.velocity = self.w*particle.velocity + self.c1*(particle.pbest-particle.path) + self.c2*(self.gbest-particle.path)
                particle.path = particle.path + particle.velocity
                
                # check pi is in the map or not
                particle.adjust_path(self.map)

                # calculate fitness function of new position
                particle.evaluate_cost(self.map, self.start, self.goal)
                particle.update_pbest()

                # update pbest, gbest
                if particle.cost < self.gbest_cost:
                    self.gbest = particle.path
                    self.gbest_cost = particle.cost

            # plot the best solution 
            # print('best sol', self.gbest)
            # print('shape of the best sol: ', self.gbest.shape)
            # plt.plot([self.start[0], self.gbest[1][0]], [self.start[1], self.gbest[1][1]], 'ro', linestyle = '--')
            # for j in range(self.gbest.shape[0]-1):
            #     plt.plot([self.gbest[j][0], self.gbest[j+1][0]], [self.gbest[j][1], self.gbest[j+1][1]], 'ro', linestyle = '--')
            # plt.plot([self.goal[0], self.gbest[-1][0]], [self.goal[1], self.gbest[-1][1]], 'ro', linestyle = '--')
            # plt.pause(0.5)
            print('Iteration ', i, ': Best cost ', self.gbest_cost)

    def create_random_path(self):
        path = []
        for i in range(self.num_waypoints):
            rand_waypoint = self.sample_a_point()
            path.append(rand_waypoint)
        return np.array(path)

    # sample a random point within map limits
    def sample_a_point(self):
        while True:
            new_x = random.uniform(0, self.map.shape[1])
            new_y = random.uniform(0, self.map.shape[0])
            if self.map[int(new_y), int(new_x)] == 0:  # Ensure point is not in an obstacle
                return np.array([new_x, new_y])


if __name__=="__main__":
    
    map = np.load('map/simple_map_1.npy')
    print('size = ', map.shape)
    start = np.array([100.0, 100.0])
    goal = np.array([500.0, 500.0])

    fig = plt.figure("PSO algorithm")
    plt.imshow(map, cmap='binary')
    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bo')
    ax = fig.gca()
    plt.xlabel('X-axis $(m)$')
    plt.ylabel('Y-axis $(m)$')

    # PSO parameters
    num_iterations = 300
    num_particles = 50
    num_waypoints = 10
    w = 1
    c1 = 1 
    c2 = 1

    pso = PSO(map, start, goal, num_iterations, num_particles, num_waypoints, w, c1, c2)
    pso.run()

    # plot the waypoints 
    complete_path = np.concatenate((start.reshape(1, -1), pso.gbest, goal.reshape(1, -1)), axis=0)
    for i in range(len(complete_path)-1):
        plt.plot([complete_path[i][0], complete_path[i+1][0]], [complete_path[i][1], complete_path[i+1][1]], 'go', linestyle = '--')
    plt.pause(0.5)
    plt.show()