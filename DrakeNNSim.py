

import torch
import torch.nn as nn

import concurrent.futures

import os
import random

from CartNN import CartNN
from PendulumSimulation import PendulumSimulation
        
def merge_state_dicts(old_state_dict, new_state_dict):
    for key, val_1 in old_state_dict.items():
        if key in new_state_dict:
            val_2 = new_state_dict[key]
            if val_1.dim() == val_2.dim():
                if val_1.dim() == 1:
                    if val_1.size(0) <= val_2.size(0):
                        new_state_dict[key][:val_1.size(0)] = val_1
            elif val_1.dim() == 2:
                if val_1.size(0) <= val_2.size(0) and val_1.size(1) <= val_2.size(1):
                    new_state_dict[key][:val_1.size(0), :val_1.size(1)] = val_1
    return new_state_dict

def change_layer(model, layer_sizes):
    chosen_index = random.randint(1, len(layer_sizes) - 1)
    chosen_layer_size = layer_sizes[chosen_index]
    layer_sizes[chosen_index] += 1 #random.randint(1, 6)
    new_model = CartNN(layer_sizes=layer_sizes)

    new_state_dict = merge_state_dicts(model.state_dict(), new_model.state_dict())

    new_model.load_state_dict(new_state_dict)

    return new_model

def add_layer(model, layer_sizes):
    chosen_index = random.randint(0, len(layer_sizes) - 1)
    layer_sizes.insert(chosen_index, 1)
    if layer_sizes[0] != 7:
        layer_sizes[0] = 7
    if layer_sizes[-1] != 1:
        layer_sizes[-1] = 1
    new_model = CartNN(layer_sizes=layer_sizes)

    new_state_dict = merge_state_dicts(model.state_dict(), new_model.state_dict())

    new_model.load_state_dict(new_state_dict)
    return new_model

def remove_layer(model, layer_sizes):
    chosen_index = random.randint(0, len(layer_sizes) - 1)
    chosen_layer_size = layer_sizes[chosen_index]
    layer_sizes.pop(chosen_index)
    if len(layer_sizes) == 1:
        layer_sizes.append(1)
    if layer_sizes[0] != 7:
        layer_sizes[0] = 7
    if layer_sizes[-1] != 1:
        layer_sizes[-1] = 1
    new_model = CartNN(layer_sizes=layer_sizes)

    new_state_dict = merge_state_dicts(model.state_dict(), new_model.state_dict())

    new_model.load_state_dict(new_state_dict)
    return new_model

def mutate(model, layer_sizes, score_stag, mutation_rate=0.05, layer_change_rate=1):
    rand_val = random.random()
    layers = []
    for i, layer in enumerate(model.model):
        layers.append(layer)
    with torch.no_grad():
        # MUTATE WEIGHTS
        if rand_val < 0.33:
            for i, param in enumerate(model.parameters()):
                if rand_val < layer_change_rate:
                    # layer_mutation_rate = mutation_rate * (1/ (i + 1))
                    mutation = torch.randn_like(param) * mutation_rate
                    param.add_(mutation)
        #ADD LAYER
        if rand_val > 0.33 and rand_val < 0.66:
            model = add_layer(model, layer_sizes)
        #CHANGE LAYER
        elif rand_val > 0.66 and rand_val < 0.95:
            if len(layers) == 1:
                model = add_layer(model, layer_sizes)
            model = change_layer(model, layer_sizes)
        #REMOVE LAYER
        elif rand_val > 0.95:
            if len(layers) > 2:
                model = remove_layer(model, layer_sizes)
    return model

def get_layer_sizes(model):
    if isinstance(model, CartNN):
        layer_sizes = []
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                layer_sizes.append(layer.in_features)
    else:
        layer_sizes = []
        for key in model.keys():
            if 'weight' in key:
                layer_size = model[key].shape[0]
                layer_sizes.append(layer_size)
            elif 'bias' in key:
                layer_size = model[key].shape[0]
                if layer_size not in layer_sizes:
                    layer_sizes.append(layer_size)
        if len(layer_sizes) > 0:
            input_size = model[next(iter(model))].shape[1]
            layer_sizes.insert(0, input_size)
    return layer_sizes

def evolve_neural_network(num_generations, population_size, duration, time_step, model_path=None):    
    gravity = 1
    drag_coefficient = 0.03
    best_score = 0
    best_model = None
    best_TOS = 0
    generation_counter = 0
    score_stag = 0
    top_models = None
    scores = []

    init_layer_sizes = [7]

    population = [CartNN(layer_sizes=init_layer_sizes) for _ in range(population_size)]

    if model_path is not None:
        saved_models_dir = model_path
        saved_model_files = [f for f in os.listdir(saved_models_dir) if f.endswith(".pth")]
        for i, model_file in enumerate(saved_model_files):
            if i >= population_size:
                break
            model_path = os.path.join(saved_models_dir, model_file)
            loaded_state_dict = torch.load(model_path)
            loaded_layer_sizes = get_layer_sizes(loaded_state_dict)[:-1]
            loaded_model = CartNN(layer_sizes=loaded_layer_sizes)
            loaded_model.load_state_dict(loaded_state_dict)
            population[i] = loaded_model

    for generation in range(num_generations):
        print(f"Generation {generation + 1} / {num_generations} with pop size: {len(population)}")

        if top_models:
            scores = top_models

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(sim.run, model, duration=duration, time_step=time_step, gravity=gravity, drag_coefficient=drag_coefficient) for model, sim in zip(population, simulations)]
            for future in concurrent.futures.as_completed(futures):
                model, score, model_time_over_threshold, total_time = future.result()
                scores.append((model, score, model_time_over_threshold))

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_TOS = model_time_over_threshold
                    best_layer_sizes = get_layer_sizes(best_model)
                    torch.save(model.state_dict(), "best_model_1.pth")
                    score_stag = 0
                    print(f"New best score: {best_score} with new best time: {round(total_time, 1)} and time over threshold: {round(best_TOS, 1)}")
                else:
                    score_stag += 1

                if model_time_over_threshold >= 9:
                    print(f"Pendulum2 stayed above height for 9s. Increasing gravity and reducing drag.")
                    gravity += 0.1
                    if gravity > 9.81:
                        gravity = 9.81
                    drag_coefficient = max(0, drag_coefficient - 0.0001)
                    break

        scores.sort(key=lambda x: x[1], reverse=True)
        for idx, (model, _, _) in enumerate(scores):
            torch.save(model.state_dict(), f"saved_models_1/model_{idx}.pth")

        top_models = sorted(scores, key=lambda x: x[1], reverse=True)[:population_size // 2]
        
        new_population = []
        for model, _, _ in top_models:
            model_layer_sizes = get_layer_sizes(model)
            mutated_model = CartNN(layer_sizes=model_layer_sizes)
            mutated_model.load_state_dict(model.state_dict())

            mutated_model = mutate(mutated_model, model_layer_sizes, score_stag)

            new_population.append(mutated_model)
        
        population = new_population
        generation_counter += 1
        print(f"Generation {generation_counter} complete with final score of {best_score} and time over threshold of {round(best_TOS, 1)}")

if __name__ == "__main__":
    populations = 32
    num_simulations = populations
    simulations = [PendulumSimulation(playback=False, enable_visualizer=False) for _ in range(num_simulations)]
    evolve_neural_network(num_generations=3000, population_size=populations, duration=10, time_step=0.1, model_path='saved_models_1')
    # evolve_neural_network(num_generations=3000, population_size=populations, duration=10, time_step=0.1, model_path=None)