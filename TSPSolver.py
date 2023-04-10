#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        # Create a full cost matrix
        edges = []
        for city in cities:
            city_edges = []
            for nextCity in cities:
                city_edges.append(city.costTo(nextCity))
            edges.append(city_edges)

        while not foundTour and time.time()-start_time < time_allowance:
            route = []
            # Now build the route using the random starting point
            route.append(random.randrange(0, ncities))

            while len(route) < ncities and time.time()-start_time < time_allowance:
                city = route[-1]
                next_city = 0
                low_cost = math.inf
                for i in range(ncities):
                    if i in route:
                        continue
                    if edges[city][i] < low_cost:
                        next_city = i
                        low_cost = edges[city][i]
                route.append(next_city)

            city_route = []
            for city in route:
                city_route.append(cities[city])
            bssf = TSPSolution(city_route)
            count += 1
            if bssf.cost < np.inf and len(route) == ncities:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        pass

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        start_time = time.time()

        # Calculate the mean x value of all the cities
        mean_x = np.mean([city._x for city in cities])

        # Split the cities into two lists based on the mean x value
        left = [city for city in cities if city._x < mean_x]
        right = [city for city in cities if city._x >= mean_x]

        # TODO: this works ok if the last point in right_solution is fairly close
        # to the first point in left_solution, but that usually isn't the case when
        # the problem size gets bigger. We'll probably need to do some reflecting/rotating
        # for one or more of the halves
        left_solution = self.hilbert_tour(16, left)
        right_solution = list(reversed(self.hilbert_tour(12, right)))
        solution = TSPSolution(left_solution + right_solution)
        return {
            'cost': solution.cost,
            'time': time.time() - start_time,
            'count': 1,
            'soln': solution,
            'max': None,
            'total': None,
            'pruned': None
        }

    def hilbert_tour(self, n, cities):
        x_range = (min(cities, key=lambda city: city._x)._x, max(cities, key=lambda city: city._x)._x)
        y_range = (min(cities, key=lambda city: city._y)._y, max(cities, key=lambda city: city._y)._y)
        mapped_cities = [
            (self.to_hilbert(n, city._x, city._y, x_range, y_range), city) for city in cities
        ]
        tour = sorted(mapped_cities, key=lambda entry: entry[0])
        return list(map(lambda x: x[1], tour))

    def to_hilbert(self, n, x, y, x_range=None, y_range=None):
        # scale x and y to be in range [0, 2^n - 1]
        if x_range is None:
            x_range = (x, x)
        if y_range is None:
            y_range = (y, y)
        x_scaled = int((x - x_range[0]) / (x_range[1] - x_range[0]) * (2 ** n - 1))
        y_scaled = int((y - y_range[0]) / (y_range[1] - y_range[0]) * (2 ** n - 1))

        # Reverse the order of the bits in the scaled coordinates
        x_binary = format(x_scaled, f'0{n}b')[::-1]
        y_binary = format(y_scaled, f'0{n}b')[::-1]

        # Interleave the bits of the binary representations of x and y
        d_binary = ''.join([b for t in zip(x_binary, y_binary) for b in t])

        # Convert the binary representation of d to an integer value
        d = int(d_binary[::-1], 2)

        return d