#####################################################################################################################################################################
#https://github.com/lmarti/evolutionary-computation-course/blob/master/AEC.03%20-%20Solving%20the%20TSP%20with%20GAs.ipynb
#https://www.youtube.com/watch?v=rGWBo0JGf50&t=1656s
#https://github.com/daydrill/ga_pycon_2016_apac/blob/master/Decision_Making_with_GA_using_DEAP.ipynb
#https://github.com/lmarti/evolutionary-computation-course/blob/master/AEC.04%20-%20Evolutionary%20Strategies%20and%20Covariance%20Matrix%20Adaptation.ipynb
#https://www.youtube.com/watch?v=uCXm6avugCo
######################################################################################################################################################################
import blackbox
oracle=blackbox.BlackBox("C:\Rud\shredded.png")
import random, operator, time, itertools, math
import numpy
import deap
import matplotlib.pyplot as plt

from deap import algorithms, base, creator, tools

def evaluation(individual):
    '''Evaluates an individual by converting it into
    a list of cities and passing that list to total_distance'''
    #return (total_distance(create_tour(individual)),)
    return ((oracle.evaluate_solution(individual)),)


alltours = itertools.permutations # The permutation function is already defined in the itertools module
cities = {1, 2, 3}
list(alltours(cities))
num_cities=30

toolbox=base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


IND_SIZE=64
toolbox = base.Toolbox()
toolbox.register("indices", numpy.random.permutation, IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list,toolbox.individual)


#toolbox.register("mate", tools.cxOrdered)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.005)
toolbox.register("evaluate", evaluation)
toolbox.register("select", tools.selTournament, tournsize=64)

pop = toolbox.population(n=100)
#print pop
fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
fit_stats.register('Min fitness', numpy.min)
fit_stats.register("Max fitness", numpy.max)
fit_stats.register("Avg. fitness", numpy.mean)
fit_stats.register("Std. deviation", numpy.std)


result, log = algorithms.eaSimple(pop, toolbox,cxpb=.9, mutpb=1,ngen=500, verbose=True,stats=fit_stats)
best_individual = tools.selBest(result, k=1)[0]
print best_individual
print('Fitness of the best individual: ', evaluation(best_individual)[0])
print oracle.show_solution(best_individual)


plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('Min fitness'),'c-', log.select('Avg. fitness'), 'b-')
plt.legend(plots, ('Minimum fitness', 'Mean fitness'), frameon=True)
plt.ylabel('Fitness'); plt.xlabel('Iterations');
plt.show()
