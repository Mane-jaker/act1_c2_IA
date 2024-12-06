import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def genetic_algorithm(dataset_path, population_size, crossover_prob, mutation_individual_prob, mutation_gene_prob, max_population, generations):
    if dataset_path.endswith('.xlsx'):
        dataset = pd.read_excel(dataset_path, skiprows=1)
    elif dataset_path.endswith('.csv'):
        dataset = pd.read_csv(dataset_path, delimiter=';')
    else:
        raise ValueError("Unsupported file format")
    
    # Elimina espacios en blanco de los nombres de las columnas
    dataset.columns = dataset.columns.str.strip()
    
    print(dataset.columns)
    
    try:
        x_values = dataset[['x1', 'x2', 'x3', 'x4', 'x5']].values.tolist()
        y_values = dataset["y"]
    except KeyError as e:
        print(f"Error: {e}")
        print("Las columnas disponibles son:", dataset.columns)
        return
    
    best_fitness = []
    worst_fitness = []
    average_fitness = []
    best_individual = []
    population = np.round(5 + np.random.rand(population_size, 6) * 5, 2).tolist()

    def plot_y(x_values, best_yc):
        plt.figure(figsize=(10, 5))
        plt.title("Objective Function vs Best Individuals")

        # Verifica el tamaño de best_yc y usa la longitud correspondiente de x_values
        x_range = range(min(len(x_values), len(best_yc)))
        plt.plot(x_range, best_yc[:len(x_range)], label="Best Individual (Yc)", color='green')

        plt.plot(range(len(x_values)), y_values, label="Objective Function (Yd)", color='red')
        plt.xlabel("Sample ID")
        plt.ylabel("Y")
        plt.legend()
        plt.show()


    def plot_fitness_evaluation(best_fitness, worst_fitness, average_fitness):
        plt.figure(figsize=(10, 5))
        plt.title("Fitness Evaluation")
        plt.plot(range(len(best_fitness)), best_fitness, label="Best", color='green')
        plt.plot(range(len(average_fitness)), average_fitness, label="Average", color='orange')
        plt.plot(range(len(worst_fitness)), worst_fitness, label="Worst", color='red')
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

    def plot_individual_parameters(best_individuals):
        plt.figure(figsize=(10, 4))
        plt.title("Parameters of the Best Individuals")
        best_a, best_b, best_c, best_d, best_e, best_f = np.transpose(best_individuals)
        plt.plot(range(len(best_individuals)), best_a, color='green', label="a")
        plt.plot(range(len(best_individuals)), best_b, color='orange', label="b")
        plt.plot(range(len(best_individuals)), best_c, color='red', label="c")
        plt.plot(range(len(best_individuals)), best_d, color='brown', label="d")
        plt.plot(range(len(best_individuals)), best_e, color='black', label="e")
        plt.plot(range(len(best_individuals)), best_f, color='blue', label="f")
        plt.xlabel("Generations")
        plt.ylabel("Parameters")
        plt.legend()
        plt.show()

    def fitness(population):
        fitness_evaluation = []
        for individual in population:
            yc_vector = []
            for x in x_values:
                y_calculated = individual[0] + individual[1]*x[0] + individual[2]*x[1] + individual[3]*x[2] + individual[4]*x[3] + + individual[5]*x[4]
                yc_vector.append(y_calculated)
            
            norm = np.linalg.norm(np.array(yc_vector) - np.array(y_values))
            fitness_evaluation.append({"individual": individual, "fitness": norm, 'Yc': yc_vector})
        return fitness_evaluation

    def select_parents(population):
        parents = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if np.random.random() <= crossover_prob:
                    parents.append((population[i], population[j]))
        return parents

    def crossover(parents):
        children = []
        for father, mother in parents:
            if len(father) > 1 and len(mother) > 1:
                child1 = [np.random.choice(genes) for genes in zip(father, mother)]
                child2 = [np.random.choice(genes) for genes in zip(father, mother)]
                children.append(child1)
                children.append(child2)
        return children

    def mutate(children):
        for i in range(len(children)):
            if np.random.random() <= mutation_individual_prob:
                for j in range(len(children[i])):
                    if np.random.random() <= mutation_gene_prob:
                        # Generar un valor aleatorio uniformemente distribuido entre -1 y 1
                        mutation_value = np.random.uniform(-1, 1)
                        children[i][j] = float(children[i][j]) + mutation_value
        return children
    
    def sort_population_by_fitness(population):
        sorted_population = sorted(population, key=lambda x: x['fitness'])
        return sorted_population

    def prune(new_population, max_population):
        # Eliminar individuos duplicados manteniendo el mejor
        unique_population = []
        seen_individuals = set()
        
        # Recorrer la población y agregar solo individuos únicos
        for individual in new_population:
            individual_tuple = tuple(individual['individual'])
            if individual_tuple not in seen_individuals:
                seen_individuals.add(individual_tuple)
                unique_population.append(individual)

        # Ordenar por fitness usando la función auxiliar
        sorted_population = sort_population_by_fitness(unique_population)

        # Seleccionar el mejor individuo
        best_individual = sorted_population[0]
        best_yc = best_individual['Yc']

        # Si la población excede el tamaño máximo, hacer la poda
        if len(sorted_population) > max_population:
            print("adios")
            rest_of_population = sorted_population[1:max_population]  # Excluir el mejor individuo y mantener hasta max_population
        else:
            print("hola")
            rest_of_population = sorted_population[1:]  # Si no excede, mantener la población tal cual

        # Construir la nueva generación incluyendo al mejor individuo
        new_generation = [best_individual['individual']] + [indiv['individual'] for indiv in rest_of_population]

        return new_generation, best_yc

    for _ in range(generations):
        parents = select_parents(population)
        children = crossover(parents)
        new_children = mutate(children)
        new_population = list(population) + list(new_children)
        fitness_evaluation = fitness(new_population)
        fitness_values = [individual['fitness'] for individual in fitness_evaluation]
        best_fitness.append(min(fitness_values))
        worst_fitness.append(max(fitness_values))
        average_fitness.append(sum(fitness_values) / len(fitness_values))
        population, best_yc = prune(fitness_evaluation,max_population)
        last_evaluation = fitness(population)
        best_individual.append(population[0])

    plot_y(x_values, best_yc)
    plot_fitness_evaluation(best_fitness, worst_fitness, average_fitness)
    plot_individual_parameters(best_individual)
    display_population_table(last_evaluation)
    
def display_population_table(final_population):
    def highlight_best_worst(table, best_indices, worst_index):
        for row in table.get_children():
            table.item(row, tags="")
        
        table.tag_configure("best", background="lightgreen")
        table.tag_configure("worst", background="lightcoral")
        
        for idx in best_indices:
            if idx < len(table.get_children()):
                table.item(table.get_children()[idx], tags=("best",))
        
        if worst_index < len(table.get_children()):
            table.item(table.get_children()[worst_index], tags=("worst",))

    population_window = tk.Toplevel()
    population_window.title("Population Table")

    columns = ("Index", "Individual", "Fitness")
    table = ttk.Treeview(population_window, columns=columns, show='headings')
    for col in columns:
        table.heading(col, text=col)
    table.grid(row=0, column=0, sticky='nsew')

    scrollbar = ttk.Scrollbar(population_window, orient=tk.VERTICAL, command=table.yview)
    table.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky='ns')

    # Ordenar la población final por fitness
    final_population.sort(key=lambda x: x['fitness'])
    
    # Obtener los índices de los mejores y peor individuos
    best_indices = list(range(5))  # Tomar los primeros 5 mejores individuos
    worst_index = len(final_population) - 1  # Tomar el último individuo (peor)

    # Insertar solo los mejores 5 y el peor individuo en la tabla
    for index in best_indices:
        individual = final_population[index]
        table.insert("", "end", values=(index, individual['individual'], individual['fitness']))

    worst_individual = final_population[worst_index]
    table.insert("", "end", values=(worst_index, worst_individual['individual'], worst_individual['fitness']))

    # Realizar el resaltado de los mejores y peores
    highlight_best_worst(table, best_indices, worst_index)




def create_gui():
    def run_algorithm():
        dataset = dataset_combobox.get()
        population_size = int(poblacion_entry.get())
        crossover_prob = float(cruce_entry.get())
        mutation_individual_prob = float(individuo_entry.get())
        mutation_gene_prob = float(genes_entry.get())
        max_population = int(max_poblacion_entry.get())
        generations = int(generaciones_entry.get())
        
        genetic_algorithm(dataset, population_size, crossover_prob, mutation_individual_prob, mutation_gene_prob, max_population, generations)

    window = tk.Tk()
    window.title("Algoritmo Genético")

    tk.Label(window, text="Ruta del Dataset:").grid(row=0, column=0, padx=10, pady=5)
    dataset_combobox = ttk.Combobox(window, values=["2024.05.22 dataset.xlsx", "213347_221216.csv", "213347_(221201).csv"], width=27)
    dataset_combobox.grid(row=0, column=1, padx=10, pady=5)
    dataset_combobox.set("Selecciona el Dataset")

    tk.Label(window, text="Tamaño Inicial de la Población:").grid(row=1, column=0, padx=10, pady=5)
    poblacion_entry = tk.Entry(window, width=10)
    poblacion_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(window, text="Probabilidad de Cruce:").grid(row=2, column=0, padx=10, pady=5)
    cruce_entry = tk.Entry(window, width=10)
    cruce_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(window, text="Probabilidad de Mutación de Individuo:").grid(row=3, column=0, padx=10, pady=5)
    individuo_entry = tk.Entry(window, width=10)
    individuo_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(window, text="Probabilidad de Mutación de Genes:").grid(row=4, column=0, padx=10, pady=5)
    genes_entry = tk.Entry(window, width=10)
    genes_entry.grid(row=4, column=1, padx=10, pady=5)

    tk.Label(window, text="Tamaño Máximo de la Población:").grid(row=5, column=0, padx=10, pady=5)
    max_poblacion_entry = tk.Entry(window, width=10)
    max_poblacion_entry.grid(row=5, column=1, padx=10, pady=5)

    tk.Label(window, text="Número de Generaciones:").grid(row=6, column=0, padx=10, pady=5)
    generaciones_entry = tk.Entry(window, width=10)
    generaciones_entry.grid(row=6, column=1, padx=10, pady=5)

    tk.Button(window, text="Ejecutar Algoritmo", command=run_algorithm).grid(row=7, column=0, columnspan=3, padx=10, pady=20)

    window.mainloop()

create_gui()
