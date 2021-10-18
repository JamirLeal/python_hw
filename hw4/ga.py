from deap import base, creator
from deap import algorithms
from deap import tools
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

w_max = 6404180
answer = 13549094
iter = 1000

# Read both datasets to get them as CSV and parse them to a pandas dataframe.
w = pd.read_csv('./p08_w.txt', header=None)
p = pd.read_csv('./p08_p.txt', header=None)

# Parse them to a Numpy Array 2D to 1D.
w = np.array(w).flatten()
p = np.array(p).flatten()

# Parse w and p to a list.
w = list(w)
p = list(p)

def evalFct(u):
    """
    En esta funcion se calcula la evaluacion y el peso. Despues se penaliza en caso de que se pase
    del peso permitido. En la tarea deben de hacer que solamente se calcule la evaluacion y que
    en otra se calcule el peso para calcular si la solucion es valida o no. Pueden usar
    tools.DeltaPenalty o la otra. Checar la lista de funciones en la libreria.

    :param u:
    :return:
    """
    profit = np.sum(np.asarray(u) * np.asarray(p))
    return profit,

def is_valid_solution(u):
    """
    Function that ensures that the total weight of the backpack is not greater than the limit.
    """
    return np.sum(np.asarray(u) * np.asarray(w)) <= w_max

# Definir si es un problema de maximizar o minimizar.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Definir que los individuos son listas y que se va a maximizar.
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Seleccionar la función de selección.
toolbox.register("select", tools.selRoulette)
# Seleccionar la función de mutación.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
# Seleccionar el de reproducción.
toolbox.register("mate", tools.cxOnePoint)
# Definir la función de evaluación.
toolbox.register("evaluate", evalFct)
# Definir la función de validez
toolbox.decorate("evaluate", tools.DeltaPenality(is_valid_solution, 0))
# Definir un elemento del individuo.
toolbox.register("attribute", random.randint, a=0, b=1)
# Definiendo la creación de individuos como una lista de n elementos.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(w))
# Definiendo la creación de la población.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

logbook = tools.Logbook()

# Save best 1000 elements found.
hof = tools.HallOfFame(1000)

# Plot best of generation.
def plot_curve(log, function_name):
    history = [log[1][i]['max'] for i in range(iter)]
    plt.figure(figsize=(20, 12))
    plt.title('Curva del mejor encontrado haciendo uso de {} con DEAP para el problema de la mochila'.format(function_name))
    plt.xlabel('Iteration')
    plt.ylabel('Best profit')
    plt.plot(history)
    plt.show()

# Plot best 1000 elements.
def plot_hof(hof, function_name):
    hof = [evalFct(i) for i in hof]
    plt.figure(figsize=(20, 12))
    plt.title('Hall of fame haciendo uso de {} con DEAP para el problema de la mochila'.format(function_name))
    plt.xlabel('Iteration')
    plt.ylabel('Best profit')
    plt.plot(hof)
    plt.show()

log = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof, cxpb=1.0, mutpb=1.0,
                    ngen=iter, stats=stats, verbose=False)

plot_curve(log, 'ea Simple')
plot_hof(hof, 'ea simple')

print('Mejor profit encontrado usando ea simple: {}'.format(evalFct(hof[1])[0]))

log = algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=5, lambda_=3, halloffame=hof, cxpb=0.5, mutpb=0.5,
                    ngen=iter, stats=stats, verbose=False)

plot_curve(log, 'eaMuPlusLambda')
plot_hof(hof, 'eaMuPlusLambda')
print('Mejor profit encontrado usando ea mu plus lambda: {}'.format(evalFct(hof[1])[0]))

log = algorithms.eaMuCommaLambda(population=pop, toolbox=toolbox, mu=5, lambda_=7, halloffame=hof, cxpb=0.5, mutpb=0.5,
                    ngen=iter, stats=stats, verbose=False)

plot_curve(log, 'eaMuCommaLambda')
plot_hof(hof, 'eaMuCommaLambda')
print('Mejor profit encontrado usando ea mu comma lambda: {}'.format(evalFct(hof[1])[0]))