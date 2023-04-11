#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT6':
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

        while not foundTour and time.time() - start_time < time_allowance:
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
        cities = self._scenario.getCities()
        num_cities = len(cities)
        pruned = 0

        start_time = time.time()

        initial_results = self.get_fancy_solution(True)
        if time.time() - start_time >= time_allowance:
            return initial_results
        bssf = initial_results['soln']

        # find initial state
        costs = np.array([
            [cities[i].costTo(cities[j]) for j in range(num_cities)] for i in range(num_cities)
        ])
        initial_state = PartialPathState(costs)
        initial_state.reduce()

        # initialize queue with initial state
        queue = [initial_state]
        max_queue_size = 1
        total_created = 1
        solutions_found = 0

        def prune_queue():
            nonlocal queue, pruned
            queue_size = len(queue)

            # prune queue items with lower bounds too high
            queue = [value for value in queue if value.lower_bound < bssf.cost]
            pruned += queue_size - len(queue)
            heapq.heapify(queue)

        while len(queue) > 0 and time.time() - start_time < time_allowance:
            max_queue_size = max([len(queue), max_queue_size])

            # pop next item from queue
            state = heapq.heappop(queue)
            current_city = state.included_cities[-1]

            # expand state popped from queue
            for dest in set(range(num_cities)) - set(state.included_cities):
                if dest == current_city:
                    continue
                total_created += 1
                child_state = state.copy()
                child_state.depth += 1
                child_state.select_and_reduce(current_city, dest)
                if child_state.lower_bound < bssf.cost:
                    if len(child_state.included_cities) == num_cities:
                        solutions_found += 1
                        bssf = TSPSolution([cities[i] for i in child_state.included_cities])
                        prune_queue()
                    else:
                        heapq.heappush(queue, child_state)
                else:
                    pruned += 1

        prune_queue()
        return {
            'cost': bssf.cost,
            'time': time.time() - start_time,
            'count': solutions_found,
            'soln': bssf,
            'max': max_queue_size,
            'total': total_created,
            'pruned': pruned
        }

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''
    def z_curve(self, time_allowance=60.0):
        return self.get_fancy_solution(False)

    def hilbert(self, time_allowance=60.0):
        return self.get_fancy_solution(True)

    def get_fancy_solution(self, hilbert=True):
        cities = self._scenario.getCities()
        start_time = time.time()

        # Calculate the mean x value of all the cities
        mean_x = np.mean([city._x for city in cities])

        # Split the cities into two lists based on the mean x value
        left = [city for city in cities if city._x < mean_x]
        right = [city for city in cities if city._x >= mean_x]

        left_sol = self.hilbert_tour(left) if hilbert else self.z_curve_tour(16, left)
        right_sol = self.hilbert_tour(right, True) if hilbert else list(reversed(self.z_curve_tour(16, right)))
        solution = TSPSolution(left_sol + right_sol)

        return {
            'cost': solution.cost,
            'time': time.time() - start_time,
            'count': 1,
            'soln': solution,
            'max': None,
            'total': None,
            'pruned': None
        }

    def get_ranges(self, cities):
        """
        get the minimum and maximum x & y values for a list of cities
        """
        range_x = (min(cities, key=lambda city: city._x)._x, max(cities, key=lambda city: city._x)._x)
        range_y = (min(cities, key=lambda city: city._y)._y, max(cities, key=lambda city: city._y)._y)
        return range_x, range_y

    def z_curve_tour(self, n, cities):
        """
        sort cities by their corresponding d-index on the z curve
        """
        range_x, range_y = self.get_ranges(cities)
        return [
            entry[1] for entry in sorted(
                [(self.map_to_z(n, city._x, city._y, range_x, range_y), city) for city in cities],
                key=lambda e: e[0]
            )
        ]

    def map_to_z(self, n, x, y, range_x, range_y):
        """
        get the corresponding d-index for a point x,y residing in range_x and range_y
        """
        # scale x and y to be in range [0, 2 ** n - 1]
        x_scaled = int((x - range_x[0]) / (range_x[1] - range_x[0]) * (2 ** n - 1))
        y_scaled = int((y - range_y[0]) / (range_y[1] - range_y[0]) * (2 ** n - 1))

        # Reverse the order of the bits in the scaled coordinates
        x_binary = format(x_scaled, f'0{n}b')[::-1]
        y_binary = format(y_scaled, f'0{n}b')[::-1]

        # Interleave the bits of the binary representations of x and y
        d_binary = ''.join([b for t in zip(x_binary, y_binary) for b in t])

        # Convert the binary representation of d to an integer value
        return int(d_binary[::-1], 2)

    def hilbert_tour(self, cities, clockwise=False):
        """
        :param cities: the cities to sort
        :param clockwise: whether to rotate the curve clockwise, if False, rotates the curve counter-clockwise
        :return: the cities sorted by their corresponding d-index on a rotated hilbert curve
        """
        range_x, range_y = self.get_ranges(cities)
        n = 2 ** 8

        def map_city(c):
            return self.map_to_hilbert(n, *self.hilbert_transform(n, c._x, c._y, range_x, range_y, clockwise)), c

        return [t[1] for t in sorted(map(map_city, cities))]

    def hilbert_transform(self, n, x, y, range_x, range_y, clockwise=False):
        """
        scale and rotate a point x,y in range_x and range_y to such that it can be
        mapped to a d-index on a rotated order-n hilbert curve
        """
        scaled_x = (x - range_x[0]) / (range_x[1] - range_x[0]) * (n - 1)
        scaled_y = (y - range_y[0]) / (range_y[1] - range_y[0]) * (n - 1)
        translated_x = scaled_x - (n - 1) / 2
        translated_y = scaled_y - (n - 1) / 2
        rotated_x = (n - 1) / 2 + (-translated_y if clockwise else translated_y)
        rotated_y = (n - 1) / 2 + (translated_x if clockwise else -translated_x)
        return int(rotated_x), int(rotated_y)

    def map_to_hilbert(self, n, x, y):
        """
        map a scaled & rotated point x,y to its corresponding d-index
        """
        rx, ry, s, d = 0, 0, 0, 0
        s = n // 2
        while s > 0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = self.hilbert_rotate(n, x, y, rx, ry)
            s //= 2
        return d

    def hilbert_rotate(self, n, x, y, rx, ry):
        """
        determine when to change direction of the n-order curve
        based on current x, y, rx, and ry
        """
        ret_x, ret_y = x, y
        if not ry:
            if rx:
                ret_x = n - 1 - x
                ret_y = n - 1 - y
            t = ret_x
            ret_x = ret_y
            ret_y = t
        return ret_x, ret_y
