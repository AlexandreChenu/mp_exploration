import random
import numpy as np
import scipy.spatial


def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[0], b[0], c[0], d[0]
    y1, y2, y3, y4 = a[1], b[1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    if denom == 0:
        return False
    else:
        t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
        if t < 0 or t > 1:
            return False
        else:
            t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
            if t < 0 or t > 1:
                return False
            else:
                return True

class Container(object):
    """ StructureGrid to approximate expansion
    """
    def __init__(self, bins_per_dim, bins_per_dim_small, dims_ranges):
        self.grid_exp = dict()

        # self.nb_corridors = 4
        # self.x_mins = [-1 + i*0.5 for i in range(0,self.nb_corridors)]
        # self.y_mins = [-1 + i*0.5 for i in range(0,self.nb_corridors)]

        self.dim = len(dims_ranges)

        self.bins_per_dim = bins_per_dim
        self.bins_per_dim_small = bins_per_dim_small

        self.dims_ranges = dims_ranges
        self.mins = np.array([r[0] for r in self.dims_ranges])
        self.maxs = np.array([r[1] for r in self.dims_ranges])


    def init_grid(self):
    # Initialize a grid with a dictionary with dimension self.dim and containing empty lists
        for i in range(self.bins_per_dim[0]): # Do not generalize to arbitrary dimensions
            for j in range(self.bins_per_dim[1]):
                self.grid_exp[(i,j)] = []

    def init_sub_grid(self, bins_per_dim):
    # Initialize a grid with a dictionary with dimension self.dim and containing empty lists
        sub_grid = dict()

        for i in range(self.bins_per_dim_small[0]): # Do not generalize to arbitrary dimensions
            for j in range(self.bins_per_dim_small[1]):
                sub_grid[(i,j)] = []

        return sub_grid


    def add(self, node):
        node = np.array(node)
        norm_node = (node - self.mins)/(self.maxs - self.mins)
        bins = np.array(norm_node*self.bins_per_dim, dtype=int)

        self.grid_exp[tuple(bins)].append(node)

    def get_expansion(self):

        n_occ = 0

        for key in self.grid_exp.keys():
            cell = self.grid_exp[key]

            if len(cell) != 0:
                n_occ += 1

        return n_occ/len(self.grid_exp)

    def get_coverage(self):

        L_coverage = []

        for key in self.grid_exp.keys():

            small_grid = self.init_sub_grid([])

            L_occupied = []

            n_total = 100 #10x10 grid

            if len(self.grid_exp[key]) != 0:
                for node in self.grid_exp[key]:
                    norm_node = (node - self.mins)/(self.maxs - self.mins)
                    bins = np.array(norm_node*self.bins_per_dim_small, dtype=int)
                    small_grid[tuple(bins)].append(node)

                for key in small_grid.keys(): #compute occupation of the subgrid
                    #print("key = ", key)
                    if len(small_grid[key]) != 0:
                        L_occupied.append(1)

                n_occupied = sum(L_occupied)

                L_coverage.append(float(n_occupied)/float(n_total))

        return sum(L_coverage)/len(L_coverage)


class Dataset(object):
    """Hold a set of vectors and provides nearest neighbors capabilities"""

    def __init__(self):
        """
        :arg dim:  the dimension of the data vectors
        """
        self.data = []
        self.dim = 2
        self.reset()

    def __repr__(self):
        return 'Databag(dim={0}, data=[{1}])'.format(self.dim, ', '.join(str(x) for x in self.data))

    def add(self, x):
        assert len(x) == self.dim
        self.data.append(x)
        self.size += 1
        self.nn_ready = False

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data     = []
        self.size     = 0
        self.kdtree   = None  # KDTree
        self.nn_ready = False # if True, the tree is up-to-date.

    def nn(self, x, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of x in the observed input data
        :arg x:      center
        :arg k:      the number of nearest neighbors to return (default: 1)
        :arg eps:    approximate nearest neighbors.
                     the k-th returned value is guaranteed to be no further than
                     (1 + eps) times the distance to the real k-th nearest neighbor.
        :arg p:      Which Minkowski p-norm to use. (default: 2, euclidean)
        :arg radius: the maximum radius (default: +inf)
        :return:     distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim, 'dimension of input {} does not match expected dimension {}.'.format(len(x), self.dim)
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(np.array(x), k_x, radius = radius, eps = eps, p = p)

    def get(self, index):
        return self.data[index]

    def iter(self):
        return iter(self.data)

    def _nn(self, v, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Compute the k nearest neighbors of v in the observed data,
        :see: nn() for arguments descriptions.
        """
        self._build_tree()
        dists, idxes = self.kdtree.query(v, k = k, distance_upper_bound = radius,
                                         eps = eps, p = p, n_jobs = -1)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self):
        """Build the KDTree for the observed data
        """
        if not self.nn_ready:
            self.kdtree   = scipy.spatial.cKDTree(self.data)
            self.nn_ready = True

    def __len__(self):
        return self.size

class Tree(object):
    def __init__(self):
        """
        Tree representation
        """
        self.V = Dataset()
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent

class RRT():
    def __init__(self, env, x_init, x_goal, budget, d_max):
        """
        Template RRT planner
        :param X: Search Space
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.env = env
        self.samples_taken = 0

        self.Grid = Container([4,4], [40,40], [[-1.001,1.001],[-1.001,1.001]])
        self.Grid.init_grid()

        self.x_init = x_init
        # self.x_goal = x_goal
        self.tree = Tree()
        self.add_vertex(self.x_init)

        self.budget = budget
        self.d_max = d_max



    def add_vertex(self, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """

        self.tree.V.add(v)
        self.tree.V._build_tree()
        self.tree.V_count += 1  # increment number of vertices in tree
        self.samples_taken += 1  # increment number of samples taken
        self.Grid.add(v)

    def add_edge(self, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        self.tree.E[child] = parent

    def get_nearest(self, x):
        """
        Return vertex nearest to x
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """

        dist, indx = self.tree.V._nn(x)
        #print("indx = ", indx)

        x_nearest = self.tree.V.get(indx[0])

        return x_nearest

    def expand(self):
        """
        Expand the search tree
        """
        x_rand = self.env.observation_space.sample() # TODO: adapt to gym environment
        x_rand = tuple(x_rand)
        # print("x_rand = ", x_rand)

        x_nearest = self.get_nearest(x_rand)
        # print("x_nearest = ", x_nearest)

        d = np.random.uniform(0., self.d_max) # random distance sampled between the sampled goal and the selected node

        x_new = self.steer(x_nearest, x_rand, d)
        # print("x_new = ", x_new)

        #print("self.tree.V.data = ", self.tree.V.data)

        # check if new point is in X_free and not already in V
        if self.tree.V.data.count(tuple(x_new)) != 0:
            return None, None

        self.samples_taken += 1
        return x_new, x_nearest

    def steer(self, start, goal, d):
        """
        Return a point in the direction of the goal, that is distance away from start
        :param start: start location
        :param goal: goal location
        :param d: distance away from start
        :return: point in the direction of the goal, distance away from start
        """

        start = np.array(start)
        goal = np.array(goal)

        v = goal - start # vector

        u = v / np.linalg.norm(v) # normalized vector

        steered_point = start + u * d

        # print("steered_point = ", steered_point)

        b_intersect = False

        for (w1, w2) in self.env.walls:
            #print("start = ", start)
            #print("steered_point = ", steered_point)
            #print("w1 = ", w1)
            #print("w2 = ", w2)

            if intersect(start, steered_point, w1, w2) is True:
                b_intersect = True

        if abs(steered_point[0]) > 1. or abs(steered_point[1]) > 1.:
            b_intersect = True

        if b_intersect == True:
            return tuple(start)
        else:
            return tuple(steered_point)


    def can_connect_to_goal(self, x_goal):
        """
        Check if the goal can be connected to the graph
        :return: True if can be added, False otherwise
        """

        x_nearest = self.get_nearest(x_goal)

        if x_goal in self.tree.E and x_nearest in self.tree.E[x_goal]:
            # tree is already connected to goal using nearest vertex
            return True

        return False

    def connect_to_goal(self, x_goal):
        """
        Connect x_goal to graph
        (does not check if this should be possible, for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        x_nearest = self.get_nearest(x_goal)
        self.tree.E[x_goal] = x_nearest

    def step(self, x_goal):

        x_new, x_nearest = self.expand()

        # print("x_new = ", x_new)
        # print("x_nearest = ", x_nearest)

        if x_new != None and x_nearest != None:
            self.add_vertex(x_new)
            self.add_edge(x_new, x_nearest)

        b = self.can_connect_to_goal(x_goal)

        if b == True:
            self.connect_to_goal(x_goal)
            return True
        else:
            return False


    def search(self, x_goal, iteration):

        #self.add_vertex(x_goal)

        success = True
        stop = False
        cnt = 0

        f_cov_exp = open("coverage_expansion_rrt_" + str(iteration) +".txt", "w")
        f_cov = open("cov_rrt_" + str(iteration) +".txt", "w")

        while stop == False:

            if cnt % 100 == 0:
                print(str(cnt), end='', flush=True)
            elif cnt %10 == 0:
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)

            stop = self.step(x_goal)
            cnt += 1

            # for node in self.tree.V.data:
            #     if node[0] > 0.8 and node[1] > 0.8:
            #         return success, cnt

            if cnt % 100 == 0:
                exp = self.Grid.get_expansion()
                cov = self.Grid.get_coverage()
                f_cov_exp.write(str(exp) + "\n")
                f_cov.write(str(cov) + "\n")

            if cnt > self.budget:
                stop = True
                success = False

        return success, cnt
