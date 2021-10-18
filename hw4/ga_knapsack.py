from deap import base, creator
from deap import algorithms
from deap import tools
import numpy as np
import random

w = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
p = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]

w_max = 165


def func_eval(u):
    """
    En esta funcion se calcula la evaluacion y el peso. Despues se penaliza en caso de que se pase
    del peso permitido. En la tarea deben de hacer que solamente se calcule la evaluacion y que
    en otra se calcule el peso para calcular si la solucion es valida o no. Pueden usar
    tools.DeltaPenalty o la otra. Checar la lista de funciones en la libreria.

    :param u:
    :return:
    """
    fu = np.sum(np.asarray(u) * np.asarray(p))
    wi = np.sum(np.asarray(u) * np.asarray(w))
    if (wi>w_max):
        fu = fu + (w_max - wi) * 5
    fu = max(fu,0)
    return fu,


# Definir si es un problema de maximizar o minimizar.
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# Definir que los individuos son listas y que se va a maximizar.
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Seleccionar la función de selección.
toolbox.register("select", tools.selRoulette)
# Seleccionar la función de mutación.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
# Seleccionar el de reproducción.
toolbox.register("mate", tools.cxOnePoint)
# Definir la función de evaluación.
toolbox.register("evaluate", func_eval)
# Definir un elemento del individuo.
toolbox.register("attribute", random.randint, a=0, b=1)
# Definiendo la creación de individuos como una lista de n elementos.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
# Definiendo la creación de la población.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof = tools.HallOfFame(3)

log = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof, cxpb=1.0, mutpb=1.0,
                    ngen=10, stats=stats, verbose=True)

print(hof)