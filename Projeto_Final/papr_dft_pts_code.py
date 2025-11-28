import numpy as np
import sionna as sn
import matplotlib.pyplot as plt
from scipy import special
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
import time
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split

NUM_BITS_PER_SYMBOL = int(input("Number of Bits per Symbol: "))
Np = int(input("Pilots Number: "))
N = int(input("Carriers Number: "))

M = 2**NUM_BITS_PER_SYMBOL 

Ntx=100000
BLOCK_LENGTH = (N-Np)*NUM_BITS_PER_SYMBOL

cont = 0

#%% Allocation of pilots and carriers:
    
def create_matrix(batch, N):    
    matrix = np.zeros((batch, N), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, N, 1)
    return matrix

def create_pilot(batch, Np):
    allocation_position = N // Np
    matrix = np.zeros((batch, Np), dtype=int)
    for i in range(batch):
        matrix[i] = np.arange(0, Np * allocation_position, allocation_position)
    return matrix

def pilot_value_change(batch, Np, pilots):
    matrix = np.zeros((batch, Np), dtype=complex)
    for i in range(batch):
        for j in range(Np): 
            matrix[i, j] = pilots[j]
    return matrix

#%% Generation of bits and modulation:
    
def Bits(batch, BL):
    binary_source = sn.utils.BinarySource()
    return binary_source([batch, BL])

def Modulation(bits):
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, normalize = True)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(bits)

#%% 
    
def symbol(x_, pilots):
    allCarriers = create_matrix(int(np.size(x_) / (N - Np)), N)
    pilotCarriers = create_pilot(int(np.size(x_) / (N - Np)), Np)
    dataCarriers = np.delete(allCarriers, pilotCarriers, axis=1)     
    symbol = np.zeros((int(np.size(x_) / (N - Np)), N), dtype=complex)
    pilots_values = pilot_value_change(int(np.size(x_) / (N - Np)), Np, pilots)
    symbol[:, pilotCarriers] = pilots_values
    symbol[np.arange(int(np.size(x_) / (N - Np)))[:, None], dataCarriers] = x_
    return symbol

def symbol_Ori(x):
    allCarriers = create_matrix(int(np.size(x)/(N-Np)), N-Np)           
    symbols = np.zeros((int(np.size(x)/(N-Np)), N-Np), dtype=complex)  # the overall N subcarriers
    symbols[np.arange(int(np.size(x)/(N-Np)))[:, None], allCarriers] = x # assign values to datacarriers
    return symbols

def FFT(symbol_):
    OFDM_time = 1/(np.sqrt(N)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFT(symbol):
    OFDM_time__ = np.sqrt(N) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time__

def FFTOri(symbol_):
    OFDM_time = 1/(np.sqrt(N-Np)) * np.fft.fft(symbol_).astype('complex64')    
    return OFDM_time
    
def IFFTOri(symbol):
    OFDM_time__ = np.sqrt(N-Np) * np.fft.ifft(symbol).astype('complex64')   
    return OFDM_time__

#%%  PAPR and CCDF:
    
def PAPR(info):
    idy = np.arange(0, info.shape[0]) 
    PAPR_red = np.zeros(len(idy)) 
    for i in idy: 
        var_red = np.mean(abs(info[i])**2) 
        peakValue_red = np.max(abs(info[i])**2) 
        PAPR_red[i] = peakValue_red / var_red

    PAPR_dB_red = 10 * np.log10(PAPR_red) 
    return PAPR_dB_red

def PAPR_DFT(info):
    var_red = np.mean(abs(info)**2)  # Potência média do sinal inteiro
    peakValue_red = np.max(abs(info)**2)  # Pico do sinal
    PAPR_dB_red = 10 * np.log10(peakValue_red / var_red)  # Retorna um escalar
    return PAPR_dB_red

def CCDF(PAPR_final):
    PAPR_Total_red = PAPR_final.size 
    mi = min(PAPR_final)
    ma = max(PAPR_final)
    eixo_x_red = np.arange(mi, ma, 0.1) 
    y_red = []
    for jj in eixo_x_red:
        A_red = len(np.where(PAPR_final > jj)[0])/PAPR_Total_red
        y_red.append(A_red)    
    CCDF_red = y_red
    return CCDF_red, eixo_x_red

#%%

bits = Bits(Ntx, BLOCK_LENGTH)
mod = Modulation(bits)
symbol__ = mod
OFDM_W_Pil = IFFT(symbol__)
Pilots_ori = []        
for Pil in range(0,Np):
    Pilots_ori.append(0)
Pilots_ori = np.array(Pilots_ori)    
symbol_Ori_ = np.zeros((bits.shape[0], N), dtype=np.complex64)
for i in range(symbol_Ori_.shape[0]):
    symbol_Ori_[i] = symbol(mod[i], Pilots_ori)    
    print(i)
#symbol_Ori_ = symbol(mod, Pilots_ori)
OFDM_time_ = IFFT(mod)
#OFDM_time_with_CP = IFFT_CP_(OFDM_time_)
_PAPR_dB = PAPR(OFDM_time_)
_CCDF, x = CCDF(_PAPR_dB)    


# Teste:
'''    
bits1 = Bits(Ntx, BLOCK_LENGTH)
mod1 = Modulation(bits1)
Pilots_ori = []        
for Pil in range(0,Np):
    Pilots_ori.append(0)
Pilots_ori = np.array(Pilots_ori)        
symbol_Ori_1 = symbol(mod1, Pilots_ori)
'''
#%% DFT

Symbol_DFT = mod

step = N // 4

P1_DFT = Symbol_DFT[:, 0:step]
P2_DFT = Symbol_DFT[:, step:2*step]
P3_DFT = Symbol_DFT[:, 2*step:3*step]
P4_DFT = Symbol_DFT[:, 3*step:N]


#P1_DFT = Symbol_DFT[:, 0:4]
#P2_DFT = Symbol_DFT[:, 4:8]
#P3_DFT = Symbol_DFT[:, 8:12]
#P4_DFT = Symbol_DFT[:, 12:16]

Pt1 = FFTOri(P1_DFT)
Pt2 = FFTOri(P2_DFT)
Pt3 = FFTOri(P3_DFT)
Pt4 = FFTOri(P4_DFT)

F_DFT = np.concatenate([Pt1, Pt2, Pt3, Pt4],axis=1)
OFDM_DFT = IFFTOri(F_DFT)

PAPR_dB_DFT = PAPR(OFDM_DFT)
    
CCDF_final_DFT, w = CCDF(PAPR_dB_DFT)    

#%% PTS

# All permutations of phase factor B
p = [0.70710677, -0.70710677, -0.70710677j, 0.70710677j]  # phase factor possible values
B = []

for b1 in range(4):
    for b2 in range(4):
        for b3 in range(4):
            for b4 in range(4):
                B.append([p[b1], p[b2], p[b3], p[b4]])  # all possible combinations
B = np.array(B)

L = 4

ofdm_symbol = symbol__

NN = Ntx
papr_min = np.zeros((NN,1))
ofdm_symbol_reconstructed = np.zeros((NN, N-Np), dtype=complex)
sig = np.zeros((NN, N-Np), dtype=complex)
PP1 = np.zeros((NN, N-Np), dtype=complex)
PP2 = np.zeros((NN, N-Np), dtype=complex)
PP3 = np.zeros((NN, N-Np), dtype=complex)
PP4 = np.zeros((NN, N-Np), dtype=complex)

a = np.zeros((NN,1), dtype=complex)
b = np.zeros((NN,1), dtype=complex)
c = np.zeros((NN,1), dtype=complex)
d = np.zeros((NN,1), dtype=complex)


ic_complex_pts = np.zeros((NN,1))

for i in range(NN):
   ic_pts = 0
   time_domain_signal = IFFTOri(ofdm_symbol[i, :]) 
   meano = np.mean(np.abs(time_domain_signal)**2)
   peako = np.max(np.abs(time_domain_signal)**2)
   papro = 10 * np.log10(peako/meano)
   
   # Partition OFDM Symbol
   
   #P1 = np.concatenate([ofdm_symbol[i, 0:4], np.zeros(10)])
   #P2 = np.concatenate([np.zeros(4), ofdm_symbol[i, 4:8], np.zeros(6)])
   #P3 = np.concatenate([np.zeros(8), ofdm_symbol[i, 8:12], np.zeros(2)])
   #P4 = np.concatenate([np.zeros(12), ofdm_symbol[i, 12:14]])
   

   P1 = np.concatenate([ofdm_symbol[i, 0:step], np.zeros((N-Np) - step)])
   P2 = np.concatenate([np.zeros(step), ofdm_symbol[i, step:2*step], np.zeros((N-Np) - 2*step)])
   P3 = np.concatenate([np.zeros(2*step), ofdm_symbol[i, 2*step:3*step], np.zeros((N-Np) - 3*step)])
   P4 = np.concatenate([np.zeros(3*step), ofdm_symbol[i, 3*step:(N-Np)]])

   Pt1 = (IFFTOri(P1))
   Pt2 = (IFFTOri(P2))
   Pt3 = (IFFTOri(P3))
   Pt4 = (IFFTOri(P4))
   
   PP1[i, :] = Pt1
   PP2[i, :] = Pt2
   PP3[i, :] = Pt3
   PP4[i, :] = Pt4
   
   papr_min[i] = papro
   print(i)
   for k in range(B.shape[0]):
       final_signal = B[k,0]*Pt1 + B[k,1]*Pt2 + B[k,2]*Pt3 + B[k,3]*Pt4 # Combination of the signal.
       meank = np.mean(np.abs(final_signal)**2)
       peak = np.max(np.abs(final_signal)**2)
       papr = 10 * np.log10(peak/meank)
       
       if papr < papr_min[i]:
                            
           a[i] = B[k, 0]
           b[i] = B[k, 1]
           c[i] = B[k, 2]
           d[i] = B[k, 3]

           papr_min[i] = papr
           sig[i, :] = final_signal
           break     
    
CCDF_PTS_re, papr_min_ = CCDF(papr_min)

#%% PAPR, Complexity, Accuracy and Loss:

fig, ax = plt.subplots(figsize=(10, 8))    
ax.semilogy(x, _CCDF, '-o', color='C2', label=f'PAPR Original', linewidth=3.5)
ax.semilogy(papr_min_, CCDF_PTS_re, 'o', color='C0', label='PTS ', linewidth=2.5)
ax.semilogy(w, CCDF_final_DFT, '<', color='C4', label='DFT', linewidth=2.5)
ax.set_xlabel('PAPR (dB)', fontsize=17, fontweight='bold')
ax.set_ylabel('CCDF', fontsize=17, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
ax.set_facecolor('white')
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_ylim([1e-3, 1])
plt.savefig('PAPR.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%% SNR x BER

class UncodedSystemAWGN(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        pilotCarriers = create_pilot(batch_size, Np)
        # Channel     
        self.OFDM_RX_FD = OFDM_
        y = self.awgn_channel([self.OFDM_RX_FD, no]) # no = potência do ruído
        #rem_CP = Remove_CP_(y)
        y_= FFTOri(y)      
        #y_without_pilots = np.delete(y_, pilotCarriers,axis=1)
        llr = self.demapper([y_,no])     
        return bits, llr
   
class UncodedSystemAWGN_PTS(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol, normalize = True)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        pilotCarriers = create_pilot(batch_size, Np)  
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)  
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        self.OFDM_RX_FD_ = sig
                
        # Channel     
        _y_ = self.awgn_channel([self.OFDM_RX_FD_, no]) # no = potência do ruído        
    
        Pt1_original = (_y_ - b * PP2 - c * PP3 - d * PP4) / a
        Pt2_original = (_y_ - a * PP1 - c * PP3 - d * PP4) / b
        Pt3_original = (_y_ - a * PP1 - b * PP2 - d * PP4) / c
        Pt4_original = (_y_ - a * PP1 - b * PP2 - c * PP3) / d
        
        F1 = FFTOri(Pt1_original)
        F2 = FFTOri(Pt2_original)
        F3 = FFTOri(Pt3_original)
        F4 = FFTOri(Pt4_original)   
        
        # Reconstruct the OFDM Symbol
        ofdm_symbol_reconstructed = np.concatenate((F1[:, 0:16], F2[:, 16:32], F3[:, 32:48], F4[:, 48:62]), axis=1)  
        
        print('PRINT PTS OUTPUT:', ofdm_symbol_reconstructed.dtype)  # Deve imprimir complex128
        print('PRINT NO PTS OUTPUT:', no.dtype)
        
        #ofdm_symbol_reconstructed = np.delete(ofdm_symbol_reconstructed, pilotCarriers,axis=1)
        llr_PTS = self.demapper([ofdm_symbol_reconstructed, no])
        #print(llr_PTS)
        self.Out_PTS = ofdm_symbol_reconstructed
        
        return bits, llr_PTS


class UncodedSystemAWGN_DFT(Model): 
    def __init__(self, num_bits_per_symbol, block_length,Subcarriers):

        super().__init__() 

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = BLOCK_LENGTH
        self.N = Subcarriers
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol, normalize = True)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        
    def __call__(self, batch_size, ebno_db):
        
        h = np.array([1])
        self.H = np.fft.fft(h,self.N)  
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        bits1 = Bits(1, self.block_length)
        mod1 = Modulation(bits)
        P1_DFT = mod1[0:4]
        P2_DFT = mod1[4:8]
        P3_DFT = mod1[8:12]
        P4_DFT = mod1[12:14]

        Pt1 = FFT(P1_DFT)
        Pt2 = FFT(P2_DFT)
        Pt3 = FFT(P3_DFT)
        Pt4 = FFT(P4_DFT)

        F_DFT = np.concatenate((Pt1, Pt2, Pt3, Pt4))
        OFDM_DFT = IFFT(F_DFT)
        
        self.OFDM_RX_FD_DFT = OFDM_DFT
        # Channel     
        _y_DFT = self.awgn_channel([self.OFDM_RX_FD_DFT, no]) # no = potência do ruído        
        
        FFT_DFT = FFTOri(_y_DFT)

        P1_DFT_rec = FFT_DFT[0:4]
        P2_DFT_rec = FFT_DFT[4:8]
        P3_DFT_rec = FFT_DFT[8:12]
        P4_DFT_rec = FFT_DFT[12:14]
        
        Pt1_rec = IFFT(P1_DFT_rec)
        Pt2_rec = IFFT(P2_DFT_rec)
        Pt3_rec = IFFT(P3_DFT_rec)
        Pt4_rec = IFFT(P4_DFT_rec)
        
        # Reconstruct the OFDM Symbol
        ofdm_symbol_DFT_reconstructed = np.concatenate((Pt1_rec, Pt2_rec, Pt3_rec, Pt4_rec))
        
        
        ofdm_symbol_DFT_reconstructed = tf.cast(ofdm_symbol_DFT_reconstructed, tf.complex64)
        no = tf.cast(no, tf.float32)
        
        print("Received OFDM:", _y_DFT)
        print("Reconstructed Symbol:", ofdm_symbol_DFT_reconstructed)

        llr_DFT = self.demapper([ofdm_symbol_DFT_reconstructed, no])
    
        return bits1, llr_DFT

  

model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                       Subcarriers=N-Np)
model_uncoded_awgn_PTS = UncodedSystemAWGN_PTS(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                               Subcarriers=N-Np)
model_uncoded_awgn_DFT = UncodedSystemAWGN_DFT(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=BLOCK_LENGTH, 
                                               Subcarriers=N-Np)

SNR = np.arange(0, 15)

EBN0_DB_MIN = min(SNR) # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = max(SNR) # Maximum value of Eb/N0 [dB] for simulations

# DFT Simulation:
    
ber_plots_DFT = sn.utils.PlotBER()
ber_plots_DFT.simulate(model_uncoded_awgn_DFT,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = 1,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_DFT = np.array(ber_plots_DFT.ber).ravel()

# Original Simulation:

ber_plots = sn.utils.PlotBER()
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM = np.array(ber_plots.ber).ravel()


# PTS Simulation:
 
ber_plots_PTS = sn.utils.PlotBER()
ber_plots_PTS.simulate(model_uncoded_awgn_PTS,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size = Ntx,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=1000, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=False);

BER_SIM_PTS = np.array(ber_plots_PTS.ber).ravel()
  
#%% Theoretical:

M = 2**(NUM_BITS_PER_SYMBOL)
L = np.sqrt(M)
mu = 4 * (L - 1) / L  # Número médio de vizinhos
Es = 3 / (L ** 2 - 1) # Fator de ajuste da constelação
    
ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
BER_THEO = np.zeros((len(ebno_dbs)))
BER_THEO_des = np.zeros((len(ebno_dbs)))

i = 0
for idx in ebno_dbs:
    BER_THEO_des[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(((N-Np)/N)*Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    BER_THEO[i] = (mu/(2*np.log2(M)))*special.erfc(np.sqrt(Es*NUM_BITS_PER_SYMBOL*10**(idx/10))/np.sqrt(2))
    i = i+1

fig, ax = plt.subplots(figsize=(10, 8)) 
ax.plot(ebno_dbs, BER_THEO, '-', label=f'Original Theoretical')
ax.plot(ebno_dbs, BER_THEO_des, '-', color='C7', label='MCSA Theoretical', linewidth=2) 
ax.plot(ebno_dbs, BER_SIM_PTS, '^', markersize=5, color='C3', label='PTS')  
ax.plot(ebno_dbs, BER_SIM_DFT, '<', markersize=5, color='C4', label='DFT Spread')    
ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, fontweight='bold')
ax.set_xlabel('Eb/N0 (dB)', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=17)
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.7, color='gray')
ax.yaxis.grid(True, which='minor', linestyle='--', alpha=0.5, color='gray')
ax.grid(axis='both', linestyle='--', alpha=0.7, color='gray')
ax.set_facecolor('white')
ax.legend(loc='upper right', fontsize=17, bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', edgecolor='black')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])
ax.set_ylim([1e-5, 1])
ax.set_yscale('log')
plt.savefig('BER.pdf', bbox_inches='tight', dpi=300)
plt.show()