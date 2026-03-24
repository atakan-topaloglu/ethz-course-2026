import zarr
import numpy as np

# Point this to the dataset you want to inspect
dataset_path = "datasets/processed/single_cube/processed_ee_xyz.zarr"
root = zarr.open(dataset_path, mode='r')
data = root['data']

print(f"\n{'Data Key [Dimension]':<28} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Median':>8} | {'Std Dev':>8}")
print("-" * 85)

# Iterate through all the arrays in the data folder
for key in sorted(data.keys()):
    arr = data[key][:]
    
    # If the array is 2D (like shape 14866, 3), we calculate stats for each column
    if len(arr.shape) == 2:
        num_dims = arr.shape[1]
        for d in range(num_dims):
            dim_data = arr[:, d]
            name = f"{key} [{d}]"
            
            # Calculate stats
            d_min = np.min(dim_data)
            d_max = np.max(dim_data)
            d_mean = np.mean(dim_data)
            d_med = np.median(dim_data)
            d_std = np.std(dim_data)
            
            print(f"{name:<28} | {d_min:8.3f} | {d_max:8.3f} | {d_mean:8.3f} | {d_med:8.3f} | {d_std:8.3f}")
            
    # If it's 1D (unlikely based on your tree, but just in case)
    else:
        dim_data = arr.flatten()
        d_min, d_max = np.min(dim_data), np.max(dim_data)
        d_mean, d_med, d_std = np.mean(dim_data), np.median(dim_data), np.std(dim_data)
        print(f"{key:<28} | {d_min:8.3f} | {d_max:8.3f} | {d_mean:8.3f} | {d_med:8.3f} | {d_std:8.3f}")

print("\n")