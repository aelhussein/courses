###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
filename = 'ps1_cow_data.txt'
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # TODO: Your code here
    f = open(filename, "r").read().splitlines()
    cows = {}
    for line in f:
        string = str.split(line, ',')
        cows[string[0]] = int(string[1])
    return cows
cows = load_cows(filename)

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    full_trip = []
    copy_cows = cows.copy()
    trip = []
    next_trip = 0
    while len(copy_cows) > 0 :
        sorted_cow = sorted(copy_cows, key = copy_cows.__getitem__,reverse = True)
        for cow in sorted_cow:
            weight = copy_cows[cow]
            if weight + next_trip <= limit:
                trip.append(cow)
                del copy_cows[cow]
                next_trip += weight
            if next_trip == limit: # don't waste time checking cows if this trip is full
                 break
        full_trip.append(trip) 
        next_trip = 0
        trip = []
    return full_trip

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    copy_cows = cows.copy()
    options = []
    total_list = []
    option_weight = 0
    for partition in get_partitions(copy_cows.keys()):
        options.append(partition)
    for option in options:
        list_options_weight = []
        for names in option:
            for name in names:
                weight = copy_cows[name]
                option_weight += weight
            list_options_weight.append(option_weight)
            option_weight = 0
        if all( x <= limit for x in list_options_weight):
            total_list.append((option,len(list_options_weight)))
    trips = []
    for i in range(len(total_list)):
        trips.append(total_list[i][1])
    index_trip = trips.index(min(trips))
    solution = total_list[index_trip][0]
    return solution
    
   
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    start = time.time()
    greedy_solution = greedy_cow_transport(cows,limit=10)
    print('greedy algo', greedy_solution, 'trips = ', len(greedy_solution))
    end = time.time()
    print(end - start)
    start = time.time()
    brute_solution = brute_force_cow_transport(cows,limit=10)
    print('brute force algo', brute_solution, 'trips = ', len(brute_solution))
    end = time.time()
    print(end - start)
    return

