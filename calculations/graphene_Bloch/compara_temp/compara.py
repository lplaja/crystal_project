import numpy as np
old = np.loadtxt("antiguo.txt")      # loadtxt ignora las líneas con '#'
new = np.loadtxt("nuevo.txt")
print(f'old.shape={old.shape}, new.shape={new.shape}')
print(old.shape == new.shape)
print(np.max(np.abs(new[2,:] - old[2,:] ), axis=0) / np.max(np.abs(old[2,:] ), axis=0))