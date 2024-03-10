import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_data(idx_list, small = False, decode = False):
    data = []
    for idx in idx_list:
        if decode: print(idx)
        if small:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
            data_matrix = data_matrix[::10]
        else:
            data_matrix = np.loadtxt(f'./datos_tfg/datos_tfg/tfg_datos_{idx[0]}_{idx[1]}.txt')
        data.append(data_matrix)
        
    return data