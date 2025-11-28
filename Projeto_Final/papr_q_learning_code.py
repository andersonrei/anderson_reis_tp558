import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import sionna as sn
from scipy import special
import tensorflow as tf
import time
from tensorflow import keras
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.initializers import HeNormal
from itertools import product

# ------------------------------- AMBIENTE ----------------------------------------------------------------
class Environment():
    def __init__(self, N_data=14, N_total=16, num_symbols=1000):
      self.N_data = N_data
      self.N_total = N_total
      self.num_symbols = num_symbols
      self.M = 4 
      self.cont = 0
      self.NUM_BITS_PER_SYMBOL = np.log2(self.M)
      self.N_pilots = 2
      #self.Eo = 1
      #self.E = (2 / 3) * (self.M - 1) * self.Eo
      self.E = 1
      self.BLOCK_LENGTH_Ori = (self.N_total) * self.NUM_BITS_PER_SYMBOL
      self.BLOCK_LENGTH = (self.N_total - self.N_pilots) * self.NUM_BITS_PER_SYMBOL
      self.binary_source = sn.utils.BinarySource()
      self.constellation = sn.mapping.Constellation("qam", self.NUM_BITS_PER_SYMBOL, normalize=True)
      self.mapper = sn.mapping.Mapper(constellation=self.constellation)
      self.demapper = sn.mapping.Demapper("app", constellation=self.constellation, hard_out=True)
      self.awgn_channel = sn.channel.AWGN()

      self.bits_total = np.array([
              self.binary_source([1, int(self.BLOCK_LENGTH)]) for _ in range(self.num_symbols)
          ])
          
          # Map each bit sequence to a symbol
      self.symbols_fixos = np.squeeze(np.array([
          self.mapper(bits) for bits in self.bits_total
      ], dtype=np.complex64))
      
      self.symbols_min = np.zeros((self.num_symbols, N_data), dtype=complex)
      self.tx_signal_total_Pil = np.zeros((self.num_symbols, N_total), dtype=complex)
      self.papr_total_Pil = np.zeros((self.num_symbols,))


    def transmitir_sem_pilotos(self, num_states, papr_total):
          # Transmissor sem pilotos
          bits = self.binary_source([papr_total.size, int(self.BLOCK_LENGTH)])
          symbols = self.mapper(bits)
          
          frame = np.zeros((papr_total.size,self.N_total), dtype=complex)
          frame[:, 1:1+self.N_data] = symbols  
          frame[:, 0] = np.sqrt(self.E)/2   # primeiro elemento 
          frame[:, -1] = np.sqrt(self.E)/2 # segundo elemento
          
          tx_signal = np.fft.ifft(frame, n=self.N_total, axis=1) * np.sqrt(self.N_total) 
          papr_w_Pil = env.calcular_papr(tx_signal)
          return bits, tx_signal, papr_w_Pil

    def transmitir_com_pilotos(self, Pil, symbol):
          # Transmissor com pilotos
          Pil = np.array(Pil)
          frame = np.zeros((self.N_total,), dtype=complex)
          frame[1:1+self.N_data] = symbol  
          frame[0] = Pil[0]   # primeiro elemento 
          frame[-1] = Pil[1]  # segundo elemento
          tx_signal = np.fft.ifft(frame, n=self.N_total) * np.sqrt(self.N_total)
          return tx_signal

    def receptor(self, tx_signal, bits_transmitidos, EbN0_dB, modo):
        #Es = 1  # Energia do símbolo (normalizada)
        EbN0 = 10**(EbN0_dB / 10)
        N0 = self.E / (np.log2(self.M) * EbN0)
    
        if modo == 'com_pilotos':
            N0 = N0 * (self.N_total / (self.N_total - self.N_pilots))
    
        N0 = tf.cast(N0, tf.float32)
    
        # Conversão para numpy array
        tx_signal = np.array(tx_signal)
        bits_transmitidos = np.array(bits_transmitidos)
    
        # Canal AWGN
        rx_signal = self.awgn_channel([tx_signal, N0])
    
        # FFT e normalização
        rx_symbols = np.fft.fft(rx_signal, n=self.N_total, axis=1) / np.sqrt(self.N_total)
        rx_symbols = rx_symbols.astype(np.complex64)
        # Seleciona apenas os dados (removendo pilotos)
        rx_symbols = rx_symbols[:, 1:1 + self.N_data]
        
        # Demapeamento para bits
        rx_bits_binary = self.demapper([rx_symbols, N0])
        rx_bits_binary = np.array(rx_bits_binary)
    
        # Debug: imprime os shapes
        print("bits_transmitidos.shape:", bits_transmitidos.shape)
        print("rx_bits_binary.shape:", rx_bits_binary.shape)
    
        # Verifica se os shapes são iguais
        if bits_transmitidos.shape != rx_bits_binary.shape:
            try:
                rx_bits_binary = rx_bits_binary.reshape(bits_transmitidos.shape)
            except:
                raise ValueError(f"Shapes incompatíveis: bits_transmitidos {bits_transmitidos.shape}, rx_bits_binary {rx_bits_binary.shape}")
    
        # BER
        ber = np.mean(bits_transmitidos != rx_bits_binary)

        return ber, bits_transmitidos, rx_bits_binary

      
    def calcular_papr(self, info):
        info = np.array(info)
        
        if info.ndim == 1:
            potencia = np.abs(info)**2
            potencia_max = np.max(potencia)
            potencia_media = np.mean(potencia)
            papr = 10 * np.log10(potencia_max / potencia_media)
            return papr
        
        else:
            idy = np.arange(0, info.shape[0]) 
            PAPR_red = np.zeros(len(idy)) 
            for i in idy: 
                potencia = np.abs(info[i])**2
                var_red = np.mean(potencia) 
                peakValue_red = np.max(potencia) 
                PAPR_red[i] = peakValue_red / var_red
    
            papr = 10 * np.log10(PAPR_red) 
            return papr


    def calcular_ccdf(self, PAPR_final):
         PAPR_final = np.array(PAPR_final).flatten()  
         PAPR_Total_red = PAPR_final.size
         eixo_x_red = np.arange(min(PAPR_final), max(PAPR_final), 0.001)
         CCDF_red = [len(np.where(PAPR_final > jj)[0]) / PAPR_Total_red for jj in eixo_x_red]
         return eixo_x_red, CCDF_red


    def plotar_ccdf(self, eixo_x1, ccdf1, eixo_x2, ccdf2):
         plt.figure(figsize=(8,6))
         plt.semilogy(eixo_x1, ccdf1, label='Sem Pilotos')
         plt.semilogy(eixo_x2, ccdf2, label='Com Pilotos')
         plt.grid(True, which='both')
         plt.xlabel('PAPR [dB]')
         plt.ylabel('CCDF')
         plt.title('Comparação CCDF - Sem e Com Pilotos')
         plt.ylim(1e-4, 1)
         plt.legend()
         plt.show()

    def step(self, visited_states, act, bits_min, symbols_min, tx_signal_total_Pil, Pilots):
         done = False
         tx_signal_pil = self.transmitir_com_pilotos(Pilots[act], self.symbol)
          
         papr_val_Pil = self.calcular_papr(tx_signal_pil)
         
         # Modo atualização da tabela:
         #limiar = 7.5
         if papr_val_Pil <= self.papr_total_Pil[self.current_index]:
            self.symbols_min[self.current_index] = self.symbol
            self.tx_signal_total_Pil[self.current_index] = tx_signal_pil
            self.papr_total_Pil[self.current_index] = papr_val_Pil
            r = 1
         else:
             r = - 1
             self.symbols_min[self.current_index] = self.symbol
             self.tx_signal_total_Pil[self.current_index] = tx_signal_pil
             self.papr_total_Pil[self.current_index] = papr_val_Pil
         
         self.current_index += 1
         self.cont += 1
          
         if self.current_index != self.num_symbols:
             next_s = self.current_index
             self.symbol = self.symbols_fixos[self.current_index]
         else:
             done = True#self.current_index == self.num_symbols
             next_s = self.current_index-1
             self.symbol = self.symbols_fixos[self.current_index-1]
             
         return Pilots, r, next_s, done

    def reset(self):
        self.current_index = 0
        self.cont = 0

        self.symbol = self.symbols_fixos[self.current_index]
        state = self.current_index
        return state

env = Environment()

#----------------------AGENTE------------------------------------------------#

def Agent(env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodios):
    # Inicializações
    num_states = 1000
    num_actions = env.M**(env.N_pilots)
    Q = np.zeros((num_states, num_actions)) 
    symbols_min = []#np.zeros((num_states, env.N_total - env.N_pilots), dtype=complex)
    tx_signal_total_Pil = []#np.zeros((num_states,env.N_total), dtype=complex)
    papr_total_Pil = []#np.zeros((num_states))
    bits_min = []#np.zeros((num_states,int(env.BLOCK_LENGTH)))
    total_rewards = []
    
    #valores_base = np.array([0.70710677, -0.70710677])
    #Pilots1 = np.random.choice(valores_base, size=num_actions) + 1j*np.random.choice(valores_base, size=num_actions)
    #Pilots2 = np.random.choice(valores_base, size=num_actions) + 1j*np.random.choice(valores_base, size=num_actions)
    # Pilotos determinísticos: todas combinações possíveis de N_pilots a partir de valores_base
    # valores base para QPSK (±1/√2)
    valores_base = np.array([0.70710677, -0.70710677], dtype=np.float32)
    #self.valores_base = np.array([0.31622777, -0.31622777, 0.9486833, -0.9486833], dtype=np.float32)
    #self.valores_base = np.linspace(0, 4*np.pi, 5, endpoint=False)
    # primeiro: gerar os 4 símbolos QPSK possíveis (combinando real x imag)

    qpsk_symbols = [complex(r, i) for (r, i) in product(valores_base, repeat=2)]
     #qpsk_symbols tem 4 elementos: [+0.707+0.707j, +0.707-0.707j, -0.707+0.707j, -0.707-0.707j]

    # agora: todas as combinações possíveis de (Pilot1, Pilot2) -> 4 x 4 = 16
    pilotos_possiveis = list(product(qpsk_symbols, repeat=env.N_pilots))

    # converter para array (forma: [16, 2]) onde cada linha é [pilot1, pilot2]
    Pilots = np.array(pilotos_possiveis, dtype=np.complex64)
    print(Pilots)
    
    #papr_total_Pil = papr_total.copy() # papr_total_Pil é uma cópia independente de papr_total.
    
    for episodio in range(episodios):
        total_rewards_por_episodio = []
        s = env.reset()
        done = False
        visited_states = set()
        while not done: 
           if np.random.rand() < epsilon:
                a = np.random.choice(num_actions)
           else:
                a = np.argmax(Q[s, :])
            
           Pilots, r, next_s, done = env.step(
                visited_states, a, bits_min, symbols_min, tx_signal_total_Pil,
                Pilots)
           
           # Atualizar Q(s,a)
           Q[s, a] += alpha * (r + gamma * np.max(Q[next_s, :]) - Q[s, a])
           #print(s)
           total_rewards.append(r)
           
           # Atualizar estado
           s = next_s
           
          
         # decaimento do epsilon
        if epsilon > epsilon_min:
           epsilon *= epsilon_dec
        print(f"Episódio {episodio+1}/{episodios} | Recompensa: {r:.3f} | Epsilon: {epsilon}")
        total_rewards_por_episodio.append(total_rewards)
        
    return num_states, bits_min, symbols_min, tx_signal_total_Pil, Q, total_rewards_por_episodio, Pilots
st = time.time()    
num_states, bits_min, symbols_min,  tx_signal_total_Pil, Q, total_rewards_por_episodio, Pilots = Agent(
    env,
    alpha=0.9,
    gamma=0.85,
    epsilon=1,
    epsilon_min=0.1,
    epsilon_dec=0.995,
    episodios=1000,
)
et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')

#%% PAPR:
  
def calcular_ccdf(PAPR_final):
    PAPR_Total_red = PAPR_final.size 
    mi = min(PAPR_final)
    ma = max(PAPR_final)
    eixo_x_red = np.arange(mi, ma, 0.1) 
    y_red = []
    for jj in eixo_x_red:
        A_red = len(np.where(PAPR_final > jj)[0])/PAPR_Total_red
        y_red.append(A_red)    
    CCDF_red = y_red
    return eixo_x_red, CCDF_red

tx_signal_total = np.zeros((env.num_symbols,env.N_total), dtype=complex)
bits_total = np.zeros((env.num_symbols,int(env.BLOCK_LENGTH)))
papr_total = np.zeros((env.num_symbols))
bits_total, tx_signal_total, papr_total = env.transmitir_sem_pilotos(env.num_symbols, papr_total)

eixo_x_w_Pil, ccdf_w_Pil = calcular_ccdf(papr_total)

# Com pilotos
eixo_x_Pil, ccdf_Pil = calcular_ccdf(env.papr_total_Pil)

env.plotar_ccdf(eixo_x_w_Pil, ccdf_w_Pil, eixo_x_Pil, ccdf_Pil)


#%% Usando a Q-Table:

papr_test = np.zeros((env.num_symbols,))


tx_signal_total_Pil_arr = np.squeeze(np.array(env.tx_signal_total_Pil, dtype=np.complex64))
symbols_min_arr = np.squeeze(np.array(env.symbols_min, dtype=np.complex64))

for ii in range(len(env.tx_signal_total_Pil)):
    # Gera bits e símbolo
    #bits = env.binary_source([1, int(env.BLOCK_LENGTH)])
    #sym = env.mapper(bits)
    #tx_signal_sem_pilotos = np.fft.ifft(symbols_min_arr[ii], n=env.N_data) * np.sqrt(env.N_data)

    estado_atual = ii

    act = np.argmax(Q[estado_atual, :])
    Pil = Pilots[act]
    #print(act)
    # Monta frame com pilotos
    frame = np.zeros(env.N_total, dtype=complex)
    frame[1:1+env.N_data] = symbols_min_arr[estado_atual]
    frame[0] = Pil[0] 
    frame[-1] = Pil[1] 
    
    tx_signal_com_pilotos = np.fft.ifft(frame, n=env.N_total) * np.sqrt(env.N_total)
    papr_final = env.calcular_papr(tx_signal_com_pilotos)

    papr_test[ii] = papr_final

eixo_test, ccdf_test = env.calcular_ccdf(papr_test)

env.plotar_ccdf(eixo_x_w_Pil, ccdf_w_Pil, eixo_test, ccdf_test)


#%% Distribuição de PAPR com e sem Q-Learning:

plt.figure(figsize=(10,6))
plt.hist(papr_total, bins=90, density=True, alpha=0.6, label='Sem Q-Learning')
plt.hist(papr_test, bins=90, density=True,alpha=0.6, label='Com Q-Learning')
plt.axvline(np.mean(papr_total), color='blue', linestyle='--', label=f'Média sem Q: {np.mean(papr_total):.2f} dB')
plt.axvline(np.mean(papr_test), color='orange', linestyle='--', label=f'Média com Q: {np.mean(papr_test):.2f} dB')
plt.xlabel('PAPR (dB)')
plt.ylabel('Densidade')
plt.title('Distribuição de PAPR com e sem Q-Learning')
plt.legend()
plt.grid(True)
plt.show()

#%% #----------------------------------BER-------------------------------------------------#

   
EbN0_dB = np.arange(0, 11)
ber_com_pilotos = []

num_iter = 30  # Número de simulações por Eb/N0

for ebn0 in EbN0_dB:
    ber2_total = 0
    for _ in range(num_iter):
        ber2, bits_transmitidos2, rx_bits2 = env.receptor(tx_signal_total_Pil, bits_min, ebn0, modo='com_pilotos')
        ber2_total += ber2
        
    ber_com_pilotos.append(ber2_total / num_iter)
    
# BER teórica:
    
    
M = 2**(env.NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
BER_THEO = np.zeros((len(EbN0_dB)))
BER_THEO_des = np.zeros((len(EbN0_dB)))

i = 0
for idx in EbN0_dB:
    BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(((env.N_total-env.N_pilots)/env.N_total)*Es*
                                                               env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*env.NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1

# Plota resultado
plt.semilogy(EbN0_dB, BER_THEO, '-', label='Original Theoretical')
plt.semilogy(EbN0_dB, BER_THEO_des, '--', label='Displaced theory')
plt.semilogy(EbN0_dB, ber_com_pilotos, 's', label='Q-Learning')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.ylim([1e-5, 1])
plt.legend()
plt.show()
