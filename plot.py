import matplotlib.pyplot as plt


def plot_mae(component_counts, maes):
    plt.figure(figsize=(10, 5))
    plt.plot(component_counts, maes, label='MAE', marker='o', color='blue')
    plt.title('MAE - sklearn')
    plt.xlabel('Number of Components')
    plt.ylabel('MAE value')
    plt.grid()
    plt.legend()
    plt.show()


def plot_rmse(component_counts, rmses):
    plt.figure(figsize=(10, 5))
    plt.plot(component_counts, rmses, label='RMSE', marker='o', color='red')
    plt.title('RMSE - sklearn')
    plt.xlabel('Number of Components')
    plt.ylabel('RMSE value')
    plt.grid()
    plt.legend()
    plt.show()