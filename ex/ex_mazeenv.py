import sys
import random
import numpy as np
from matplotlib import collections as mc
import matplotlib.pyplot as plt

class Coverage_grid(object):
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

class ExBufferElement:
    def __init__(self, state, uniqid, feature, extra=None):
        self.state = state
        self.uniqid = uniqid
        self.feature = feature
        self.extra = extra
        #self.useCounter = 0

    def __hash__(self):
        return hash(tuple(self.state)) ^ hash(self.uniqid)
    def __eq__(self,other):
        # print("self.state = ", self.state)
        # print("other.state = ", other.state)
        #
        # print("self.uniqid = ", self.uniqid)
        # print("other.uniqid = ", other.uniqid)

        # return (self.state==other.state).all() and (self.uniqid==other.uniqid).all()
        return (self.state==other.state).all() and (self.uniqid==other.uniqid)
    def __str__(self):
        return "EBE({})".format(str(self.state))
    def __repr__(self):
        return "EBE({})".format(str(self.state))


class ExBufferBin:
    def __init__(self):
        self.elements = dict()  # Use counter -> list of elements
        self.leastCounter = 0

    def insert(self, bufferElement: ExBufferElement):
        if not 0 in self.elements:
            self.elements[0] = []
        self.elements[0].append(bufferElement)
        self.leastCounter = 0

    def size(self):
        return sum([len(bucket) for bucket in self.elements])

    def minCounter(self):
        return min(self.elements.keys())

    def maxCounter(self):
        return max(self.elements.keys())

    def useLeastUsed(self, stochastic=False):
        if stochastic:
            chosen = random.choice(self.elements[self.leastCounter])
            self.elements[self.leastCounter].remove(chosen)
        else:
            chosen = self.elements[self.leastCounter].pop()
        #print("Choosing elt {} with counter={}".format(chosen,self.leastCounter))
        if not self.leastCounter+1 in self.elements:
            self.elements[self.leastCounter+1] = []
        self.elements[self.leastCounter+1].append(chosen)
        if self.elements[self.leastCounter] == []:
            del self.elements[self.leastCounter]
            self.leastCounter += 1
        return chosen

    def __str__(self):
        return "EBB({})".format(str(self.elements))
    def __repr__(self):
        return "EBB({})".format(str(self.elements))

class Ex:
    def __init__(self, env, path, feature = lambda x : x, resolution = 0.1, stochastic=True, budget= 200):
        self.resolution = resolution
        self.feature = feature
        self.stochastic = stochastic
        self.nonce = 0
        self.geo = dict()       # bin coords -> Use counter
        self.elements = dict()  # Use counter -> bin coords -> bin
        self.leastCounter = 0
        self.numElements = 0
        self.verbose = False
        self.transitions = []
        self.budget = budget
        self.env = env
        self.path = path

        self.Grid = Coverage_grid([50,50], [40,40], [[-0.01,15.01],[-0.01,15.01]])
        self.Grid.init_grid()

    def binCoordsState(self, state):
        feature = self.feature(state)
        return self.binCoordsFeature(feature)

    def binCoordsFeature(self, feature):
        return tuple(int(coord//self.resolution) for coord in feature)

    def insert(self, state, extra = None):
        self.nonce += 1
        feature = self.feature(state)
        coords = self.binCoordsFeature(feature)
        elt = ExBufferElement(state, self.nonce, feature)
        if coords in self.geo:  # Bin exists
            if self.verbose:
                print("Inserting {} in existing bin at coords {}".format(state,coords))
            exBin = self.elements[self.geo[coords]][coords]
        else:
            if self.verbose:
                print("Inserting {} in new bin at coords {}".format(state,coords))
            self.leastCounter = 0
            exBin = ExBufferBin()
            self.geo[coords] = 0
            if not 0 in self.elements:
                self.elements[0] = dict()
            self.elements[0][coords] = exBin
        exBin.insert(elt)
        self.numElements += 1

    def useLeastUsed(self):
        if self.verbose:
            print()
        dct = self.elements[self.leastCounter]
        if self.stochastic:
            coords = random.choice(list(dct.keys()))
        else:
            for coords in dct:
                break
        if self.verbose:
            print("Choosing bin {} with counter={}".format(coords,self.leastCounter))
        chosen = dct.pop(coords)
        if not self.leastCounter+1 in self.elements:
            self.elements[self.leastCounter+1] = dict()
        self.elements[self.leastCounter+1][coords] = chosen
        self.geo[coords] = self.leastCounter+1
        if self.elements[self.leastCounter] == dict():
            del self.elements[self.leastCounter]
            self.leastCounter += 1
        return chosen.useLeastUsed(stochastic=self.stochastic)

    def __str__(self):
        s = 'EX(\n  geo: {},\n  counters: '.format(self.geo)
        for n,x in enumerate(self.elements):
            if n>0:
                s += '            '
            s += '{}: {}\n'.format(x,self.elements[x])
        s += ')'
        return s

    def stats(self):
        numBins = len(self.geo)
        minBinCount = min(self.elements.keys())
        maxBinCount = max(self.elements.keys())
        minEltCount = min([min([bn.minCounter() for bn in binList.values()])
            for binList in self.elements.values()])
        maxEltCount = max([max([bn.maxCounter() for bn in binList.values()])
            for binList in self.elements.values()])
        #numElements = sum([sum([bn.size() for bn in binList]) for binList in self.elements.items()])
        s = 'EX state : {} bins, with counts from {} to {}. {} states, with counts from {} to {}' \
            .format(numBins, minBinCount, maxBinCount, self.numElements, minEltCount, maxEltCount)
        return s

    def step(self, x_goal):

        s = self.useLeastUsed().state

        self.env.state = s

        v = np.random.uniform([-0.1,-0.1],[0.1,0.1])

        b_valid = False
        #### check action is valid
        b_valid = self.env.valid_action(v)

        if b_valid == True:
            sp = s + v
            self.Grid.add(sp)

        else:
            sp = s

        if (self.binCoordsState(sp) == self.binCoordsState(s)) and False:
            print("s' is in same bin as s, not inserting this transition")
        else:
            self.insert(sp)

        self.transitions.append([s,sp])

        b = self.can_connect_to_goal(x_goal)

        if b == True:
            self.connect_to_goal(x_goal)
            return True
        else:
            return False

    def can_connect_to_goal(self, goal_state):
        # TODO
        return False

    def connect_to_goal(self, goal_state):
        # TODO
        return None

    def search(self, x_goal):

        #self.add_vertex(x_goal)

        success = True
        stop = False
        cnt = 0

        f_cov_exp = open(self.path + "/coverage_expansion_ex.txt", "w")
        # f_cov = open("cov_ex_" + str(iteration) +".txt", "w")

        while stop == False:

            if cnt % 100 == 0:
                print(str(cnt), end='', flush=True)
            elif cnt %10 == 0:
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)

            stop = self.step(x_goal)
            cnt += 1

            # print("transitions = ", self.transitions)

            # for node in self.tree.V.data:
            #     if node[0] > 0.8 and node[1] > 0.8:
            #         return success, cnt

            if cnt % 100 == 0:
                exp = self.Grid.get_expansion()
                # cov = self.Grid.get_coverage()
                f_cov_exp.write(str(exp) + "\n")
            #     f_cov.write(str(cov) + "\n")
            #
            if cnt > self.budget:
                stop = True
                success = False

        return success, cnt


if __name__=="__main__":
    ex = Ex(resolution=.1, stochastic=True)
    ex.insert([5.,5.])
    transitions = []
    for i in range(10000):
        #print(ex)
        s = ex.useLeastUsed().state
        sp = s + np.random.uniform([-0.2,-0.1],[.2,.1])
        sp = np.clip(sp, [0,0],[10,10])
        if (ex.binCoordsState(sp) == ex.binCoordsState(s)) and False:
            print("s' is in same bin as s, not inserting this transition")
        else:
            ex.insert(sp)
        transitions.append([s,sp])

    lc = mc.LineCollection(transitions)
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

    print(ex.useLeastUsed())

    sys.exit(0)
