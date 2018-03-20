import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

TYPE_0 = 0
TYPE_1 = 1
A = 10

# With constraint: x_i in [-5, 5]
def fitness_0(n, x):
    result = 0.0
    for i in range(n):
        result += x[i] * x[i]

    return result

# With constraint: x_i in [-5, 5]
def fitness_1(A, n, x):
    result = A * n
    for i in range(n):
        result += (x[i] * x[i] - A * math.cos(2 * math.pi * x[i]))

    return result


# individuals
class indivdual:
    def __init__(self):
        self.x = []  # code
        self.fitness = 0  # fitness

    def __eq__(self, other):
        self.x = other.x
        self.fitness = other.fitness

    def __repr__(self):
        return repr((self.fitness, self.x))


def evolution(N, lam, mu, type_fitness, Neval, ub, lb):
    filename = "test.txt"
    f = open(filename, "w")


    # table of fitness
    fitnesses = []

    # array of mean distances
    mean_distances = []

    # array of mean fitnesses
    mean_fitnesses = []

    # initiation of parents
    pop = []
    for i in range(mu):
        ind = indivdual()
        for j in range(N):
            ind.x.append(np.random.uniform(ub, lb))
        if type_fitness == TYPE_0:
            ind.fitness = fitness_0(N, ind.x)
        elif type_fitness == TYPE_1:
            ind.fitness = fitness_1(A, N, ind.x)
        pop.append(ind)

    # get the best one
    minFitness = 0
    index = 0
    for i in range(mu):
        if pop[i].fitness < minFitness:
            minFitness = pop[i].fitness
            index = i

    best = pop[index]

    # get the mean vector
    mean = indivdual()
    for i in range(N):
        mean.x.append(0)

    for i in range(mu):
        for j in range(N):
            mean.x[j] = mean.x[j] + pop[i].x[j]

    for i in range(N):
        mean.x[i] = mean.x[i]/mu

    if type_fitness == TYPE_0:
        mean.fitness = fitness_0(N, mean.x)
    elif type_fitness == TYPE_1:
        mean.fitness = fitness_1(A, N, mean.x)

    # get the variance
    a = []
    for i in range(mu):
        a.append(pop[i].x)
    delta = np.std(a)

    iter = 0
    #delta = 0.1
    parent = mean
    while(iter < Neval):

        print("iteration "+str(iter)+" runs...")


        # create the children and calculate their fitness
        children = []
        for i in range(lam):
            child = indivdual()
            for j in range(N):
                child.x.append(parent.x[j] + delta * rd.normalvariate(0,1))
                if child.x[j] > lb:
                    child.x[j] = child.x[j] - (lb - ub)
                if child.x[j] < ub:
                    child.x[j] = child.x[j] + (lb - ub)

            if type_fitness == TYPE_0:
                child.fitness = fitness_0(N, child.x)
            elif type_fitness == TYPE_1:
                child.fitness = fitness_1(A, N, child.x)

            children.append(child)

        # combine the lambda and mu individuals
        population = []
        for k in range(len(pop)):
            population.append(pop[k])
        for k in range(len(children)):
            population.append(children[k])

        # sort the population by fitness to choose mu individuals
        population = sorted(population, key=lambda individual: individual.fitness)
        pop = []
        for i in range(mu):
            pop.append(population[i])

        # re-calculate the best parent
        minFitness = 0
        index = 0
        for i in range(mu):
            if pop[i].fitness < minFitness:
                minFitness = pop[i].fitness
                index = i
        best = pop[index]

        # re-calculate the mean vector
        mean = indivdual()
        for i in range(N):
            mean.x.append(0)

        for i in range(mu):
            for j in range(N):
                mean.x[j] = mean.x[j] + pop[i].x[j]

        for i in range(N):
            mean.x[i] = mean.x[i] / mu
        if type_fitness == TYPE_0:
            mean.fitness = fitness_0(N, mean.x)
        elif type_fitness == TYPE_1:
            mean.fitness = fitness_1(A, N, mean.x)

        # afficher le resultat de chaque pas
        print("best fit found : " + str(best.fitness))
        fitnesses.append(best.fitness)

        # affecter le parent
        parent = mean
        mean_fitnesses.append(mean.fitness)

        # calculate the mean distance
        distances = []
        for i in range(len(pop)):
            for j in range(len(pop)):
                distance = 0.0
                for k in range(N):
                    distance += (pop[i].x[k] - pop[j].x[k])**2
                distances.append(distance)
        mean_distance = np.mean(distances) / 2
        mean_distances.append(mean_distance)

        # write in file
        f.write(str(iter) + " " + str(best.fitness) + " " + str(mean.fitness) + " " + str(mean_distance) + "\n")

        # get the variance
        a = []
        for i in range(mu):
            a.append(pop[i].x)
        delta = np.std(a)
        #delta = 0.1
        iter = iter + 1

    f.close()

    return [best, best.fitness, fitnesses, mean_fitnesses, mean_distances]


if __name__ == '__main__':
    """results = []
    i = 0
    while(i < 20):
        x_best, fit_best, result = evolution(30, 20, 15, TYPE_1, 20000, -5, 5)
        print("best individual : " + str(x_best.x))
        print("best fitness is : " + str(fit_best))
        results.append(fit_best)
        i = i + 1
    moyen = np.mean(results)
    ecart = np.std(results)
    print("La moyen des fitnesses est: " + str(moyen))
    print("L'ecart type est: " + str(ecart))"""

    x_best, fit_best, result, mean_fit, mean_dis = evolution(30, 30, 20, TYPE_1, 5000, -5, 5)
    print("best individual : " + str(x_best.x))
    print("best fitness is : " + str(fit_best))
    x = range(5000)
    plt.figure()
    plt.plot(x, result)
    plt.plot(x, mean_fit)
    plt.plot(x, mean_dis)
    plt.show()



