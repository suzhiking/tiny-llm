import numpy as np

# Basic: load a .npy file into a NumPy array
arr = np.load("data/encoded_data/encoded_tinystories.npy")   # accepts str or Path
arr2 = np.load("data/encoded_data/encoded_tinystories_python.npy")   # accepts str or Path
if np.array_equal(arr, arr2):
    print("Two files are the same")
else:
    diff = np.argwhere(arr != arr2)
    print("Different at indices:", diff)
    for i in diff[:10]:
        print(i, arr[tuple(i)], arr2[tuple(i)])
# print(arr)                          # prints the contents
# print(arr2)         # extra info
