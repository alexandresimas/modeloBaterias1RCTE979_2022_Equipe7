import numpy as np

# Definição do formato de uma resposta de um sistema com 1 RC
def RC_1(x, V0:float, V1:float, T1:float):
    y = V0 + V1*np.exp(-x/T1)
    return y

# Definição do formato de uma resposta de um sistema com 2 RCs
def RC_2(x, V0:float, V1:float, V2:float, T1:float, T2:float):
    y = V0 + V1*np.exp(-x/T1) + V2*np.exp(-x/T2)
    return y


# Definição do formato de uma resposta de um sistema com 3 RCs
def RC_3(x, V0:float, V1:float, V2:float, V3:float, T1:float, T2:float, T3:float):
    y = V0 + V1*np.exp(-x/T1) + V2*np.exp(-x/T2) + V3*np.exp(-x/T3)
    return y

# Implementação do algoritmo MSE para 2 {np.array}s
def MSE (A:np.array, B:np.array):
    mse = ((A-B)**2).mean()
    return mse