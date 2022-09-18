# bibliotecas utilizadas
import bisect as bs
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.integrate import odeint
from scipy import interpolate


# valores supostos da bateria
CapAh = 70
CapAs = CapAh * 3600

# Leitura dos valores do OCV vs SoC
df = pd.read_excel('Narada 70 Ah Pb-C OCV vs Soc.xlsx')
df.drop([0, 1, 3], inplace=True)
df.drop(columns={'Status', 'Prog Time'}, inplace=True)

# tratamento dos dados do tempo
t = df['Time Stamp']
t = t.tolist()
d = [datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p') for x in t]
t_secs = [(x - d[0]).total_seconds() for x in d]
df['Current'] = df['Current']/120.0 # normalização
d_aux = [0]

# integral da carga pelo tempo
for x in range(len(t_secs)-1):
    d_aux.append(d_aux[x] + df['Current'].values[x] * (t_secs[x+1] - t_secs[x]))

# valor de corte dos dados
min_daux = min(d_aux)
index_min = d_aux.index(min_daux)


V_SoC = d_aux[0:index_min]
t_secs = t_secs[0:index_min]
volt = df['Voltage'].values[0:index_min]

# criação dos vetores da Carga e do SoC
Carga = [(x - min_daux)/(3600.0) for x in V_SoC]
max_carga = max(Carga)
SoC = [(x*100.0/max_carga) for x in Carga]

'''
plt.plot(t_secs, V_SoC)
plt.show()
plt.plot(t_secs, Carga)
plt.show()

'''

# Interpolação linear da OCV
ocv = interpolate.interp1d(SoC, volt, fill_value="extrapolate")

# aquisição dos dados dos parâmetros
df_dados = pd.read_excel('Narada 70 Ah Pb-C parametros.xlsx')
df_dados.drop([0, 1], inplace = True)

# Corrente, Tensão e Tempo dos parâmetros
i_dados = df_dados['Current'].values
v_dados = df_dados['Voltage'].values
t = df_dados['Time Stamp']
t = t.values - t.values[0]
t = t.tolist()
t = [x.total_seconds() for x in t]

# Interpolação dos valores medidos (corrente e tensão)
f_it = interpolate.interp1d(t, i_dados, fill_value="extrapolate")
f_vt = interpolate.interp1d(t, v_dados, fill_value="extrapolate")

# Dados adquiridos no arquivo 'final_v3.py'
teeste = pd.read_csv('Teeste.csv')

v_soc = teeste.values[0]
v_r0 = teeste.values[1]
v_r1 = teeste.values[2]
v_c1 = teeste.values[3]

# interpolação dos dados encontrados de R0, R1 e C1
f_r0 = interpolate.interp1d(v_soc, v_r0, fill_value="extrapolate")
f_r1 = interpolate.interp1d(v_soc, v_r1, fill_value="extrapolate")
f_c1 = interpolate.interp1d(v_soc, v_c1, fill_value="extrapolate")

# Tratamento de bug
t[0] = 0.001
t[1] = 0.002

#
# definição das funções utilizadas mais pra frente no código
#

def i(t):
    return f_it(t)

# SoC (100 -> 100% ; 0 -> 0%)
def stateofcharge(t):
    if isinstance(t, float):
        return 100
    else:
        return 100 + (integrate.cumtrapz(i(t), t, initial=0.0)/(0.01*CapAs))

def c1(t):
    return f_c1(stateofcharge(t))

def r0(t):
    return f_r0(stateofcharge(t))

def r1(t):
    return f_r1(stateofcharge(t))

def OCV(t):
    return ocv(stateofcharge(t))


# função auxiliar para o calculo do IR1
def hlp_ir1(ir1, t, _i, _r1, _c1):
    dir1dt = (-ir1 + _i(t)) / (_r1(t) * _c1(t))
    return dir1dt

# calculo do ir1
aux = odeint(hlp_ir1, 0, t, args=(i, r1, c1))
aux2 = [0 if np.isnan(x[0]) else x[0] for x in aux]
ir1 = interpolate.interp1d(t, aux2)

# definição da tensão de saída baseada nos valores estimados
def v_certo(t):
    return OCV(t) + (i(t)*r0(t) + ir1(t)*r1(t))


plt.plot(t, v_certo(t), 'r', label='Estimado')
plt.plot(t, f_vt(t), 'b', label='Verdadeiro')
plt.legend()
plt.show()
