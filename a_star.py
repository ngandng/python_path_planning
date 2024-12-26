import heapq
import numpy as np
import matplotlib.pyplot as plt

from utils import map_to_grid, grid_point_to_map

# convert map into grid
cell_size = 20

class Node():
    def __init__(self, x, y):
        # position of the node
        self.x = int(x)
        self.y = int(y)
        self.parent = None

        # cost of the node
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.x==other.x and self.y==other.y
    def __lt__(self, other):
        return self.f < other.f  # Compare based on cost
    
class MinQueue:
    def __init__(self):
        self.queue = []
        
    def push(self, new_node:Node, new_cost):
        update = False
        # if node is in the queue, just update the value
        for i, (cost, node) in enumerate(self.queue):
            if node == new_node:
                if cost > new_cost:
                    self.queue.remove(self.queue[i]) 
                    heapq.heappush(self.queue,(new_cost, new_node))  # Update cost
                update = True
                break
        # if not add it to the queue
        if not update:
            heapq.heappush(self.queue,(new_cost, new_node))

    def pop(self):
        if self.queue:
            return heapq.heappop(self.queue)
        raise IndexError("pop from an empty queue")

    def peek(self):
        if self.queue:
            return self.queue[0][1]
        raise IndexError("peek from an empty queue")

    def is_empty(self):
        return len(self.queue) == 0        

# A-star class
class AStar():
    def __init__(self, start, goal, grid):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])

        self.grid = grid

        self.open_list = MinQueue()
        self.open_list.push(self.start, self.start.f)
        self.close_list = []
        self.current_node = None

        self.num_waypoints = 0
        self.waypoints = []
        self.path_distance = 0

    def search(self):
        while not self.open_list.is_empty():
            # let the current node is the smallest node in the open list
            self.current_node = self.open_list.pop()[1]
            # add it to the close list
            self.close_list.append(self.current_node)

            plt.pause(0.10)
            print('current grid cell ({}, {})'.format(self.current_node.x,self.current_node.y))
            current_position = grid_point_to_map(cell_size, cell_size, [self.current_node.x, self.current_node.y])
            plt.plot(current_position[0], current_position[1], 'ro')

            # if the current node is the goal, we are done
            if self.current_node == self.goal:
                self.goal.parent = self.current_node.parent
                print("We found the goal!!!")
                self.path_distance = self.current_node.g
                break
            
            # find all the children for the current node
            children = self.find_child(self.current_node.x, self.current_node.y)

            for child in children:
                
                child.parent = self.current_node
                # check whether this position is in closed list
                is_pass = False
                for element in self.close_list:
                    if child==element:
                        is_pass = True
                        break
                if is_pass:
                    continue
                child.g = self.current_node.g + 1
                child.h = self.distance(child, self.goal)
                child.f = child.h + child.g

                self.open_list.push(child, child.f)
        self.extract_path(self.goal)
        print('Number of waypoints: ', self.num_waypoints)
        print('Path distance (m): ', self.path_distance)
        print('Waypoints: ', self.waypoints)

    def find_child(self, x, y):
        children = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_child = [int(x)+dx, int(y)+dy]
                
                # check feasible location for the children
                # note that the grid is (y,x)
                if self.grid[new_child[0]][new_child[1]]==1:
                    continue
                if new_child[0] > self.grid.shape[0] or new_child[1] > self.grid.shape[1] or new_child[0]<0 or new_child[1]<0:
                    continue

                new_node = Node(new_child[0], new_child[1])
                children.append(new_node)
        return children
    
    def distance(self, a:Node, b:Node):
        return np.linalg.norm([a.x-b.x, a.y-b.y])
    
    def extract_path(self, goal):
        self.num_waypoints += 1

        current_point = np.array([goal.x, goal.y])
        self.waypoints.insert(0, current_point)
        if (goal.parent):
            self.extract_path(goal.parent)

if __name__=="__main__":
    map = np.load('map/simple_map.npy')
    print('size = ', map.shape)
    start = np.array([100.0, 100.0])
    goal = np.array([750.0, 750.0])
    
    goal_region = plt.Circle((goal[0], goal[1]), 2, color='b', fill=False)

    fig = plt.figure("A-star algorithm")
    plt.imshow(map, cmap='binary')
    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bo')
    ax = fig.gca()
    ax.add_patch(goal_region)
    plt.xlabel('X-axis $(m)$')
    plt.ylabel('Y-axis $(m)$')

    new_start = [start[0]//cell_size,start[1]//cell_size]
    new_goal = [goal[0]//cell_size,goal[1]//cell_size]
    print('new start:', new_start, 'new goal:', new_goal)
    grid = map_to_grid(cell_size, cell_size, map)

    # a_star = AStar(new_start, new_goal, grid)
    a_star = AStar(new_start, new_goal, grid)
    a_star.search()

    # plot the waypoints    
    waypoints = []
    waypoints.append(grid_point_to_map(cell_size, cell_size, a_star.waypoints[0]))
    for i in range(len(a_star.waypoints)-1):
        wp1 = grid_point_to_map(cell_size, cell_size, a_star.waypoints[i])
        wp2 = grid_point_to_map(cell_size, cell_size, a_star.waypoints[i+1])
        waypoints.append(wp2)
        plt.plot([wp1[0], wp2[0]], [wp1[1], wp2[1]], 'go', linestyle = '--')
        plt.pause(0.1)
    print('Actual Waypoints: ', waypoints)
    plt.show()