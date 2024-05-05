import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import tensorflow as tf


def load_parameter_list():
    #Loading list of parameters
    parameters = []

    for i in range(10):
        file_params = f'./datos_tfg/datos_tfg/tfg_parametros_{i+1}.txt'
        with open(file_params, 'r') as file:
            for line in file:
                numbers = line.strip().split('\t')
                numbers = [float(num) for num in numbers if num.strip()]
                arrays = [numbers[i:i+7] for i in range(0, len(numbers), 7)]
                parameters.extend(arrays)
                
    return parameters

def visualize(idx_list, parameters, cols = 2, file = None):
    
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
    if file:
        plt.savefig(file)
    plt.show()

    

def visualize_parameter_distributions(df_params, y_limit, file = None):
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
  if file:
        plt.savefig(file)
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

def r_squared(y_true_slice, y_pred_slice):
    numerator = tf.reduce_sum(tf.square(tf.subtract(y_true_slice, tf.reduce_mean(y_true_slice))))
    denominator = tf.reduce_sum(tf.square(tf.subtract(y_true_slice, y_pred_slice)))
    return 1 - tf.math.divide_no_nan(numerator, denominator + tf.keras.backend.epsilon())

def rmse(y_true_slice, y_pred_slice):
    mse = tf.keras.losses.mean_squared_error(y_true_slice, y_pred_slice) 
    return tf.sqrt(mse + tf.keras.backend.epsilon())

def plot_metrics(history, metric_name,lim_y_bottom = 0, lim_y_top = 1.6, file = None):
    plt.plot(history.history[metric_name], color='navy')
    plt.plot(history.history[f'val_{metric_name}'], color='red')
    plt.title(f'Train and validation {metric_name} during training')
    plt.ylabel(metric_name.upper())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)  
    plt.xlim(left=0)
    plt.ylim(bottom=lim_y_bottom)  
    plt.ylim(top=lim_y_top)  
    plt.savefig(file)
    plt.show()

def model_evaluation(model, X_test, y_test, random_samples = 5):
    metrics = model.evaluate(X_test, y_test)
    print('Test mse:', metrics[0])
    print('Test rmse:', metrics[1])
    print('Test mae:', metrics[2])
    print('Test R2:', metrics[3])


    # Separate ground truth and predictions for the last two outputs
    y_true_e1 = y_test['e1']
    y_true_e2 = y_test['e2']
    predictions = model.predict(X_test)
    predictions_e1 = predictions[:, -2]
    predictions_e2 = predictions[:, -1]

    # Calculate metrics independently for each output
    #e1
    mse_e1 = tf.keras.metrics.mean_squared_error(y_true_e1, predictions_e1)
    rmse_e1 = rmse(y_true_e1, predictions_e1)
    mae_e1 = tf.keras.metrics.mean_squared_error(y_true_e1, predictions_e1)
    r2_e1 = r_squared(y_true_e1, predictions_e1)

    mse_e2 = tf.keras.metrics.mean_squared_error(y_true_e2, predictions_e2)
    rmse_e2 = rmse(y_true_e2, predictions_e2)
    mae_e2 = tf.keras.metrics.mean_squared_error(y_true_e1, predictions_e2)
    r2_e2 = r_squared(y_true_e2, predictions_e2)

    print("Metrics for parameters e1 and e2:")
    print(f"Parameter e1:\n  \tMSE: {mse_e1:.3f}, RMSE: {rmse_e1:.3f}, MAE: {mae_e1:.3f}, R2: {r2_e1:.3f}")
    print(f"Parameter e2:\n  \tMSE: {mse_e2:.3f}, RMSE: {rmse_e2:.3f}, MAE: {mae_e2:.3f}, R2: {r2_e2:.3f}")


    random_indices = np.random.choice(len(y_test), size=random_samples, replace=False)
    test_samples = X_test[random_indices]
    predictions = model.predict(test_samples)
    print('\nRandom prediction examples')
    print('Parameters:\tcx\tcy\ta\tb\ttheta\te1\te2')
    for i in range(len(predictions)):
        print('------------')
        print('real:\t\t', '\t'.join(f'{val:.3f}' for val in y_test.iloc[random_indices[i]]))
        print('prediction:\t', '\t'.join(f'{val:.3f}' for val in predictions[i]))

