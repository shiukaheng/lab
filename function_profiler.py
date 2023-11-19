import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class FunctionProfiler:
    def __init__(self):
        self.profiles = {}

    def profile(self, name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time

                if name not in self.profiles:
                    self.profiles[name] = []
                self.profiles[name].append(elapsed_time)

                return result

            return wrapper

        return decorator

    def plot(self, name):
        if name not in self.profiles:
            print(f"No data found for function '{name}'")
            return

        plt.figure(figsize=(8, 6))
        plt.title(f"Kernel Density Plot for Function '{name}'")
        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Density")

        data = np.array(self.profiles[name])
        sns.kdeplot(data, color='b', shade=True)
        plt.show()

    def plot_all(self, *names):
        plt.figure(figsize=(8, 6))
        plt.title("Kernel Density Plots for Functions")
        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Density")

        for name in names:
            if name in self.profiles:
                data = np.array(self.profiles[name])
                sns.kdeplot(data, shade=True, label=name)

        plt.legend()
        plt.show()