import numpy as np
import math
import matplotlib.pyplot as plt
import random

from utils import load_img

class Node():
    def __init__(self, x, y):
        self.x = x                  # x coordinate of the node
        self.y = y                  # y coordinate of the node
        self. children = []         # children list of the node
        self.parent = None          # parent node reference

# RRT algorithm class
class RrtAlgorithm():
    def __init__(self, start, goal, numIterations, grid, stepSize):
        self.randomTree = Node(start[0], start[1])      # the root node
        self.goal = Node(goal[0], goal[1])              # goal node
        self.nearest_node = None                             
        self.iterations = min(numIterations, 2000)
        self.grid = grid                                # the map
        self.rho = stepSize
        self.path_distance = 0
        self.nearest_dist = np.inf
        self.num_way_points = 0
        self.way_points = []     

    # add the point to the nearest node and add goal when reached
    def add_child(self, x, y):
        if (x == self.goal.x):
            # add the goal node to the children of the nearest node
            self.nearest_node.children.append(self.goal)
            self.goal.parent = self.nearest_node
        else:
            temp_node = Node(x,y)
            self.nearest_node.children.append(temp_node)
            temp_node.parent = self.nearest_node

    # sample a random point within grid limits
    def sample_a_point(self):
        new_x = random.uniform(0, self.grid.shape[1])
        new_y = random.uniform(0, self.grid.shape[0])

        return np.array([new_x, new_y])

    # steer a distance stepsize from start the end location
    def steer_to_point(self, start, end):
        offset = self.rho*self.unit_vector(start, end)
        point = np.array([start.x + offset[0], start.y+offset[1]])
        # note that the map is (y,x)
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1]-1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0]-1
        return point

    # check if obstacles lies between the start node and the end point of the edge
    def is_in_obstacle(self, start, end):
        unit_vec = self.unit_vector(start, end)
        test_point = np.array([0, 0])
        for i in range(self.rho):
            test_point[0] = start.x + i*unit_vec[0]
            test_point[1] = start.y + i*unit_vec[1]
            # if grid[round(test_point[0]).astype(np.int64)][round(test_point[1]).astype(np.int64)] == 1:
            if grid[math.floor(test_point[1])][math.floor(test_point[0])] == 1:
                return True   
        return False

    # find unit vector between two points which form a vector
    def unit_vector(self, start, end):
        v = np.array([end[0]-start.x, end[1]-start.y])
        unit_vec = (v)/(np.linalg.norm(v))
        return unit_vec

    # find the nearest node from a given unconnected point (Euclidean distance)
    def find_nearest(self, root, point):
        # return condition if the root is NULL
        if not root:
            return None
        # find distance between root and point
        dist = self.distance(root, point)
        # if this is lower than the nearest distance, set this as the nearest node and update nearest distance
        if dist <= self.nearest_dist:
            self.nearest_node = root
            self.nearest_dist = dist
        # recursively call by iterating though the children
        for child in root.children:
            self.find_nearest(child, point)
    
    # dinf euclidean distance between a node and an XY point
    def distance(self, node1, point):
        return np.sqrt((point[0]-node1.x)**2 + (point[1]-node1.y)**2)

    # check if the goal had been reached with step size
    def goal_found(self, point):
        return self.distance(self.goal,point) <= self.rho

    # reset nearest_node and nearest_distance
    def reset_nearest_values(self):
        self.nearest_dist = np.inf
        self.nearest_node = None

    # trace the path from goal to start
    def retrace_rrt_path(self, goal):
        if goal.x == self.randomTree.x:
            return
        self.num_way_points += 1
        # insert currentPoint to the Waypoint array from the begining
        current_point = np.array([goal.x, goal.y])
        self.way_points.insert(0, current_point)
        self.path_distance += self.rho
        if (goal.parent):
            self.retrace_rrt_path(goal.parent)

    def run(self):
        for i in range(self.iterations):
            # reset nearest value
            self.reset_nearest_values()
            print('Iteration: ', i)
            sample_point = self.sample_a_point()
            self.find_nearest(self.randomTree, sample_point)
            new = self.steer_to_point(self.nearest_node, sample_point)
            bool = self.is_in_obstacle(self.nearest_node, new)
            if (bool == False):
                self.add_child(new[0], new[1])
                plt.pause(0.10)
                plt.plot([self.nearest_node.x, new[0]], [self.nearest_node.y, new[1]], 'go', linestyle="--")
                # if goal found, append to path
                if (self.goal_found(new)):
                    self.add_child(goal[0], goal[1])
                    print('Goal found!')
                    break

        self.retrace_rrt_path(self.goal)
        self.way_points.insert(0, [self.randomTree.x,self.randomTree.y])
        print('Number of waypoints: ', self.num_way_points)
        print('Path distance (m): ', self.path_distance)
        print('Waypoints: ', self.way_points)

if __name__=="__main__":
    # load_img('img/simple_map.png')
    # read_img('map.npy')

    grid = np.load('simple_map.npy')
    print('size = ', grid.shape)
    start = np.array([100.0, 100.0])
    goal = np.array([750.0, 750.0])
    num_iteration = 500
    step_size = 50
    goal_region = plt.Circle((goal[0], goal[1]), step_size, color='b', fill=False)

    fig = plt.figure("RRT algorithm")
    plt.imshow(grid, cmap='binary')
    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bo')
    ax = fig.gca()
    ax.add_patch(goal_region)
    plt.xlabel('X-axis $(m)$')
    plt.ylabel('Y-axis $(m)$')

    rrt = RrtAlgorithm(start, goal, num_iteration, grid, step_size)
    rrt.run()

    # plot the waypoints 
    for i in range(len(rrt.way_points)-1):
        plt.plot([rrt.way_points[i][0], rrt.way_points[i+1][0]], [rrt.way_points[i][1], rrt.way_points[i+1][1]], 'ro', linestyle = '--')
        plt.pause(1.0)

    plt.show()