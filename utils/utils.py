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
    
    
    n_rows = len(idx_list)// cols
    if len(idx_list)%cols > 0:
        n_rows += 1
    fig, axs = plt.subplots(n_rows, cols, figsize = (cols*9,n_rows*9))
    
    axes = axs.flatten()

    for i,ax in enumerate(axes):
        if i >= len(idx_list):
            axes[i].axis('off')
        else:
            file_path = file_paths[i]
            data_matrix = np.loadtxt(file_path)
            ax.imshow(data_matrix, cmap='viridis', aspect='auto')
            params = parameters[parameters['data_index'] == (idx_list[i][0], idx_list[i][1])]
            params_str = '\n'.join([f'{col}: {params.iloc[0][col]}' for col in params.columns[:-1]])
            ax.set_title(f'Case {idx_list[i][0]}_{idx_list[i][1]}\n{params_str}')
        
    
    plt.tight_layout()
    plt.show()

def visualize_parameter_distributions(df_params, y_limit):
  fig, axes = plt.subplots(2, 4, figsize=(12, 8))

  axes = axes.flatten()

  custom_x_ranges = [(-3.2,4.4), (0.5,7), (0,4.1),(0,2.5), (-1,1.5), (0,11.1),(0,12.6)]
  for i, col in enumerate(df_params.columns[:-1]):
      sns.histplot(df_params[col], ax=axes[i])
      axes[i].set_xlabel(f'{df_params.columns[i]} values')
      axes[i].set_ylabel(f'Total count') 
      axes[i].set_xlim(custom_x_ranges[i][0], custom_x_ranges[i][1])
      axes[i].set_ylim(0, y_limit) 
      axes[i].grid(True)
      #axes[i].set_title(col)

  axes[-1].axis('off')
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
        results.append((idx,data_matrix))

def load_data_multithreaded(idx_list, threads_n, small=False, debug=False):
    #NOT WORKING
    num_threads = threads_n  
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

    ordered_results = sorted(results, key=lambda x: (x[0][0], x[0][1]))
    cleaned_results  = [tupl[1] for tupl in ordered_results]
    return cleaned_results


def plot_metrics(history, metric_name):
    plt.plot(history.history[metric_name])
    plt.plot(history.history[f'val_{metric_name}'])
    plt.title(metric_name.capitalize())
    plt.ylabel(metric_name.upper())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def model_evaluation(model, X_test, y_test, random_samples = 5):
    metrics = model.evaluate(X_test, y_test)
    print('Test mse:', metrics[0])
    print('Test rmse:', metrics[1])
    print('Test mae:', metrics[2])
    print('Test r2score:', metrics[3])
    random_indices = np.random.choice(len(y_test), size=random_samples, replace=False)
    test_samples = X_test[random_indices]
    predictions = model.predict(test_samples)
    print('\n\nRandom prediction examples')
    print('Parameters:\tcx\tcy\ta\tb\ttheta\te1\te2')
    for i in range(len(predictions)):
        print('------------')
        print('real:\t\t', '\t'.join(f'{val:.3f}' for val in y_test.iloc[random_indices[i]]))
        print('prediction:\t', '\t'.join(f'{val:.3f}' for val in predictions[i]))