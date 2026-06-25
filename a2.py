import numpy as np 

def a2_categorical(t : int):
    rng = np.random.default_rng()
    a = []
    for _ in range(t):
        i = rng.random()
        a.append(1 if i <= 0.42 else (2 if i <= 0.66 else 3))
    return np.array(a)

print(np.bincount(a2_categorical(1000)))