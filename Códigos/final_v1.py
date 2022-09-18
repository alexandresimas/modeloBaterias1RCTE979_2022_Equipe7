## Bibliotecas Usadas
import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt

## Variáveis usadas nessa simulação

t_inicio = 0 # em Segundos
t_fim = 100 # em Segundos

Capacidade_Ah = 100
Capacidade_As = Capacidade_Ah*3600

iBase = 100
r0 = 700*0.000001 # em Ohms
r1 = 0.5*0.001 # em Ohms
c1 = 15000 # em Farads

# Função que retorna o estado de carga da bateria em relação ao tempo decorrido e ao comportamento da corrente
# Parte do pressuposto que o estado de carga começa em 1 (100%)
def stateofcharge(vet_it, t):
    return 1 - ((integrate.cumtrapz(vet_it(t), t, initial=0.0)) / Capacidade_As)

# Função que retorna a OCV, dependendo do SoC atual.
# Função linear, retornando 10 quando SoC = 0%, e 12 quando SoC = 100%
def OCV(state):
    return 10 + (12-10)*(state)

# Definição do comportamento da corrente -> ou 0 ou iBase 
def it(t):
    if t > t_inicio + (t_fim-t_inicio)*0.1 and t < t_inicio + (t_fim-t_inicio)*0.8:
        return 0*iBase
    return 1*iBase

# Funçao auxiliar para resolução da EDO do calculo da IR1
# Retorna a derivada de IR1 em relação ao tempo
def mod_ir1(ir1, t, r1, c1, it):
    val = 1/(r1*c1)
    dir1dt =  val*it(t) - val*ir1
    return dir1dt




# Criação do vetor tempo e os valores de it(t)
t = np.linspace(t_inicio, t_fim)
vect_it = np.vectorize(it)
plt.plot(t, vect_it(t))
plt.show()



# Criação do vetor de SoC
SoC = stateofcharge(vect_it, t)
plt.plot(t, SoC)
plt.show()
plt.plot(t, OCV(SoC))
plt.show()


# Resolução da EDO, para encontrar os valores de IR1 ao longo do tempo
ir1 = odeint(mod_ir1, 0, t, args=(r1, c1, it))
ir1 = ir1.flatten()
plt.plot(t, ir1)
plt.show()



# Criação do vetor de v(t), baseado na corrente, estado de carga, e componentes internos do sistema
vt = OCV(SoC) - (ir1*r1 + r0*vect_it(t))
plt.plot(t, vt)
plt.show()

