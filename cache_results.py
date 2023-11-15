import functools
import pickle
import os

def cache_results(func):
    @functools.wraps(func)
    def wrapper_cached(*args, **kwargs):
        # Create a unique file name based on the function name and arguments
        file_name = f"{func.__name__}_cache.pkl"

        # Check if the results are already cached
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                cached_result = pickle.load(f)
                return cached_result

        # Execute the function and cache the result
        result = func(*args, **kwargs)
        with open(file_name, 'wb') as f:
            pickle.dump(result, f)

        return result

    return wrapper_cached