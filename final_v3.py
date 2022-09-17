# Bibliotecas usadas
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from equacoes_final import *


# Leitura do arquivo de parâmetros
df = pd.read_excel('Narada 70 Ah Pb-C parametros.xlsx')

# Retirar a primeira linha, que mostra as unidades dos dados
# Retirar a segunda linha também, já que ambos os valores estão em zero, ou seja, os sensores de corrente e tensão não estao conectados.
df.drop([0, 1], inplace=True)


# Definição dos valores das localizações iniciais e finais dos dados
indexes_current_zero = df['Current'].where(df['Current']==0).dropna().index
df_index = pd.DataFrame(indexes_current_zero)

df_index_last_zeros = df_index.where((df_index.values != df_index.shift(-1)-1)).dropna()
df_index_last_zeros.drop(0, inplace=True)

df_index_first_zeros = df_index.where((df_index.values != df_index.shift(1)+1)).dropna()
df_index_first_zeros.drop(0, inplace=True)

# variável auxiliar - corte de valores iniciais
a = 1

# Valores encontrados analisando os dados - simbolizam as curvas de relaxação de 100% a 0% a cada 10%
amostras = [18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

vet_R0 = []
vet_R1 = []
vet_C1 = []


for amostra in amostras:
    # Retirar os valores de corrente, tensão e tempo
    amps = df['Current'].loc[int(df_index_first_zeros.values[amostra])-a:int(df_index_last_zeros.values[amostra])-1]
    voltz = df['Voltage'].loc[int(df_index_first_zeros.values[amostra])-a:int(df_index_last_zeros.values[amostra])-1]
    t_1 = df['Time Stamp'].loc[int(df_index_first_zeros.values[amostra])-a:int(df_index_last_zeros.values[amostra])-1]

    # Normalização da tensão
    voltz = voltz.values - voltz.values[0]
    
    # Normalização e transformaçao do vetor de tempo
    t_1 = t_1.values - t_1.values[0]
    t_1 = t_1.tolist()
    t_1 = [t_1[x].total_seconds() for x in range(len(t_1))]

    
    # tentativa de fitting da curva
    p1, _ = curve_fit(RC_1, t_1, voltz, p0=[0.1, 0.1, 10000], maxfev = 40000)

    # retirada dos valores da tupla de retorno
    v0 = float(p1[0])
    v1 = float(p1[1])
    tau1 = float(p1[2])

    print('\n\n\n')
    print(f'Valores para fitting usando modelo 1 RC e amostra {amostra}:')
    print(f'V0 = {v0:.03f} V')
    print(f'V1 = {v1:.03f} V')
    print(f'T1 = {tau1:.03f} s')

    # Calculo da tensão de saída estimada
    voltz_estimados = RC_1(np.dot(t_1, 1.0), v0, v1, tau1)

    # Calculo do MSE entre a saída medida e a saída estimada
    ms1 = MSE(np.array(voltz), np.array(voltz_estimados))

    # Calculo dos valores de R0, R1 e C1
    last_amp = df['Current'].loc[int(df_index_first_zeros.values[amostra])-a-1]
    r0 = abs(v0/last_amp)
    r1 = abs(v1/last_amp)
    c1 = tau1/r1

    # Adicionar os valores ao vetor final de valores
    vet_R0.append(r0)
    vet_R1.append(r1)
    vet_C1.append(c1)

    '''
    plt.plot(t_1, voltz, 'o', label='data')
    plt.plot(t_1, voltz_estimados, '-', label='fit')
    plt.xlabel('Amostras [s]')
    plt.ylabel('Tensão [V]')
    plt.title(f'Fitting usando 1 RC - Amostra {amostra} ---> MSE = {ms1:.03e}')
    plt.legend()
    plt.show()
    '''


# plots dos vetores encontrados
t = np.arange(100, -10, -10)
plt.figure(1)

plt.subplot(311)
plt.plot(t, vet_R0)
plt.xlabel('SoC [%]')
plt.ylabel('Resistência R0 [\u03A9]')
plt.xlim(max(t), min(t))

plt.subplot(312)
plt.plot(t, vet_R1)
plt.xlabel('SoC [%]')
plt.ylabel('Resistência R1 [\u03A9]')
plt.xlim(max(t), min(t))

plt.subplot(313)
plt.plot(t, vet_C1)
plt.xlabel('SoC [%]')
plt.ylabel('Capacitância C1 [F]')
plt.xlim(max(t), min(t))

plt.show()


vet_R0.reverse()
vet_R1.reverse()
vet_C1.reverse()

t = np.arange(0, 110, 10)

# exportar os valores encontrados pelo fitting
aux = pd.DataFrame([t, vet_R0, vet_R1, vet_C1])
aux.to_csv("Teeste.csv", index=None)


######### A partir daqui são só testes #########


t_a = np.linspace(0, 100, 1000)

plt.subplot(311)
plt.plot(t, vet_R0)
z = np.polyfit(t, vet_R0, 8)
r0_poli = np.poly1d(z)
plt.plot(t, vet_R0, '.', t_a, r0_poli(t_a), '-')

plt.subplot(312)
plt.plot(t, vet_R1)
z = np.polyfit(t, vet_R1, 8)
r1_poli = np.poly1d(z)
plt.plot(t, vet_R1, '.', t_a, r1_poli(t_a), '-')

plt.subplot(313)
plt.plot(t, vet_C1)
z = np.polyfit(t, vet_C1, 8)
c1_poli = np.poly1d(z)
plt.plot(t, vet_C1, '.', t_a, c1_poli(t_a), '-')

plt.show()





plt.subplot(311)
plt.plot(t, vet_R0)
f = interpolate.interp1d(t, vet_R0)
r0_poli = f(t_a)
plt.plot(t, vet_R0, '.', t_a, r0_poli, '-')

plt.subplot(312)
plt.plot(t, vet_R1)
f = interpolate.interp1d(t, vet_R1)
r1_poli = f(t_a)
plt.plot(t, vet_R1, '.', t_a, r1_poli, '-')

plt.subplot(313)
plt.plot(t, vet_C1)
f = interpolate.interp1d(t, vet_C1)
c1_poli = f(t_a)
plt.plot(t, vet_C1, '.', t_a, c1_poli, '-')

plt.show()
