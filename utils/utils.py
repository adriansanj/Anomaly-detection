import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import threading

def load_parameter_list():
    #Loading list of parameters
    parameters = []

    for i in range(5):
        file_params = f'./datos_tfg/datos_tfg/tfg_parametros_{i+1}.txt'
        with open(file_params, 'r') as file:
            for line in file:
                numbers = line.strip().split('\t')
                numbers = [float(num) for num in numbers if num.strip()]
                arrays = [numbers[i:i+7] for i in range(0, len(numbers), 7)]
                parameters.extend(arrays)
                
    return parameters

def visualize(idx_list, parameters, cols = 2):
    
    file_paths = [f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt' for idx in idx_list]
    params = [parameters[idx[0]-1] for idx in idx_list]  #CHANGE AND ADD PARAMETER VISUALIZATION
    
    n_rows = len(idx_list)// cols
    if len(idx_list)%cols > 0:
        n_rows += 1
    fig, axs = plt.subplots(n_rows, cols, figsize = (cols*5,n_rows*5))
    
    axes = axs.flatten()

    for i,ax in enumerate(axes):
        if i >= len(idx_list):
            axes[i].axis('off')
        else:
            file_path = file_paths[i]
            data_matrix = np.loadtxt(file_path)
            ax.imshow(data_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'Case {idx_list[i][0]}_{idx_list[i][1]}')
        
    
    plt.tight_layout()
    plt.show()

def save_indexes(idx_list, file_path):
    with open(file_path, 'w') as file:
        for item in idx_list:
            file.write(f"{item}\n")

def read_indexes(file_path):
    # Initialize an empty list to store the tuples
    read_data = []

    # Read the data from the text file
    with open(file_path, 'r') as file:
        for line in file:
            # Remove the newline character and convert the string representation of the tuple back to a tuple
            tuple_str = line.strip()
            tuple_data = eval(tuple_str)
            read_data.append(tuple_data)
    
    return read_data

def load_data(idx_list, small = False, debug = False):
    data = []
    for idx in idx_list:
        if debug: print(idx)
        if small:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
            data_matrix = data_matrix[::10]
        else:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
        data.append(data_matrix)
        
    return data

def load_data_thread(idx_list, results, small=False, debug=False):
    for idx in idx_list:
        if debug:
            print(idx)
        if small:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
            data_matrix = data_matrix[::10]
        else:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
        results.append(data_matrix)

def load_data_multithreaded(idx_list, small=False, debug=False):
    num_threads = 8  
    chunk_size = (len(idx_list) + num_threads - 1) // num_threads 
    threads = []
    results = []

    for i in range(0, len(idx_list), chunk_size):
        chunk = idx_list[i:i + chunk_size]
        thread = threading.Thread(target=load_data_thread, args=(chunk, results, small, debug))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return results


def plot_metrics(history, metric_name):
    plt.plot(history.history[metric_name])
    plt.plot(history.history[f'val_{metric_name}'])
    plt.title(metric_name.capitalize())
    plt.ylabel(metric_name.upper())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

# Model evaluation
def model_evaluation(model, X_test, y_test, random_samples = 5):
    loss, mae = model.evaluate(X_test, y_test)
    print('Test mse:', loss)
    print('Test mae:', mae)
    random_indices = np.random.choice(len(y_test), size=random_samples, replace=False)
    test_samples = X_test[random_indices]
    predictions = model.predict(test_samples)
    print('\n\nRandom prediction examples')
    print('Parameters:\tcx\tcy\ta\tb\ttheta\te1\te2')
    for i in range(len(predictions)):
        print('------------')
        print('real:\t\t', '\t'.join(f'{val:.4f}' for val in y_test.iloc[random_indices[i]]))
        print('prediction:\t', '\t'.join(f'{val:.4f}' for val in predictions[i]))