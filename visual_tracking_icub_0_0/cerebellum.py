import nest
import logging

__author__ = 'Alberto Antonietti, Edoardo Negri, Stefano Nardo'

logger = logging.getLogger(__name__)

try:
    nest.Install("albertomodule")
    logger.info("Albertomodule installed correctly")
except Exception as e:  # DynamicModuleManagementError
    logger.info(e)
    logger.info("Albertomodule already installed")

# CEREBELLUM
LTP1 = 0.05 #0.05 #0.01
LTD1 = -0.7 #-0.4 #-0.5
Init_PFPC = 4.0

LTP2 = 1e-5 #1e-5
LTD2 = -1e-6#-1e-6
Init_MFDCN = 0.07 #0.07

LTP3 = 1e-7
LTD3 = 1e-6
Init_PCDCN = -0.5 #-0.5

PLAST1 = True   # PF-PC ex
PLAST2 = False   # MF-DCN ex
PLAST3 = False   # PC-DCN in

nest.SetKernelStatus({'resolution' : 1.0})

"""
Initializes NEST with the neuronal network that has to be simulated
"""
nest.CopyModel('iaf_cond_exp', 'granular_neuron')
nest.CopyModel('iaf_cond_exp', 'purkinje_neuron')
nest.CopyModel('iaf_cond_exp', 'olivary_neuron')
nest.CopyModel('iaf_cond_exp', 'nuclear_neuron')

nest.SetDefaults('granular_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})

nest.SetDefaults('purkinje_neuron', {'t_ref': 2.0,
                                        'C_m': 400.0,
                                        'V_th': -52.0,
                                        'V_reset': -70.0,
                                        'g_L': 16.0,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 1.6})
                                    #'I_e': 300.0})

nest.SetDefaults('nuclear_neuron', {'t_ref': 1.0,
                                    'C_m': 2.0,
                                    'V_th': -40.0,
                                    'V_reset': -70.0,
                                    'g_L': 0.2,
                                    'tau_syn_ex': 0.5,
                                    'tau_syn_in': 10.0})

# Cell numbers
MF_num = 100
GR_num = MF_num*20
PC_num = 72
IO_num = PC_num
DCN_num = PC_num/2


MF = nest.Create("parrot_neuron", MF_num)
PC = nest.Create("purkinje_neuron", PC_num)
IO = nest.Create("parrot_neuron", IO_num)
DCN = nest.Create("nuclear_neuron", DCN_num)
GR = nest.Create("granular_neuron", GR_num)
vt = nest.Create("volume_transmitter_alberto",PC_num)
for n, vti in enumerate(vt):
    nest.SetStatus([vti], {"vt_num": n})
        
logger.info('MF: ' + str(min(MF)) + " " + str(max(MF)))
logger.info('GR: ' + str(min(GR)) + " " + str(max(GR)))
logger.info('PC: ' + str(min(PC)) + " " + str(max(PC)))
logger.info('IO: ' + str(min(IO)) + " " + str(max(IO)))
logger.info('DCN: ' + str(min(DCN)) + " " + str(max(DCN)))
logger.info('vt: ' + str(min(vt)) + " " + str(max(vt)))


# Connectivity

# MF-GR excitatory connections
MFGR_conn_param = {"model": "static_synapse",
                    "weight": {'distribution' : 'uniform', 'low': 0.55, 'high': 0.7},
                    "delay": 1.0}
nest.Connect(MF,GR,{'rule': 'fixed_indegree', 'indegree': 4, "multapses": False}, MFGR_conn_param)

if PLAST1:
    # PF-PC excitatory plastic connections
    # each PC receives the random 80% of the GR
    nest.SetDefaults('stdp_synapse_sinexp',
                    {"A_minus":   LTD1,
                    "A_plus":    LTP1,
                    "Wmin":      0.0,
                    "Wmax":      4.0,
                    "vt":        vt[0]})
    
    PFPC_conn_param = {"model":  'stdp_synapse_sinexp',
                    "weight": Init_PFPC,
                    "delay":  1.0}
    for i, PCi in enumerate(PC):
        nest.Connect(GR, [PCi], {'rule': 'fixed_indegree',
                                'indegree': int(0.8*GR_num),
                                "multapses": False},
                    PFPC_conn_param)
        A = nest.GetConnections(GR, [PCi])
        nest.SetStatus(A, {'vt_num': i})
        
    nest.Connect(IO, vt, {'rule': 'one_to_one'},
                        {"model": "static_synapse",
                        "weight": 1.0, "delay": 1.0})
else:
    PFPC_conn_param = {"model":  'static_synapse',
                        "weight": Init_PFPC,
                        "delay":  1.0}

    for i, PCi in enumerate(PC):
        nest.Connect(GR, [PCi], {'rule': 'fixed_indegree',
                                'indegree': int(0.8*GR_num),
                                "multapses": False},
                                PFPC_conn_param)
        
# MF-DCN excitatory connections
if PLAST2:
    vt2 = nest.Create("volume_transmitter_alberto", DCN_num)
    for n, vti in enumerate(vt2):
        nest.SetStatus([vti], {"vt_num": n})
if PLAST2:
    # MF-DCN excitatory plastic connections
    # every MF is connected with every DCN
    nest.SetDefaults('stdp_synapse_cosexp',
                            {"A_minus":   LTD2,
                            "A_plus":    LTP2,
                            "Wmin":      0.0,
                            "Wmax":      0.25,
                            "vt":        vt2[0]})
    MFDCN_conn_param = {"model": 'stdp_synapse_cosexp',
                        "weight": Init_MFDCN,
                        "delay": 10.0}
    for i, DCNi in enumerate(DCN):
        nest.Connect(MF, [DCNi], 'all_to_all', MFDCN_conn_param)
        A = nest.GetConnections(MF, [DCNi])
        # nest.SetStatus(A, {'vt_num': float(i)})
        nest.SetStatus(A, {'vt_num': i})
else:
    MFDCN_conn_param = {"model":  "static_synapse",
                        "weight": Init_MFDCN,
                        "delay":  10.0}
    nest.Connect(MF, DCN, 'all_to_all', MFDCN_conn_param)                        

# PC-DCN inhibitory plastic connections
# each DCN receives 2 connections from 2 contiguous PC
if PLAST3:
    nest.SetDefaults('stdp_synapse', {"tau_plus": 30.0,
                                            "lambda": LTP3,
                                            "alpha": LTD3/LTP3,
                                            "mu_plus": 0.0,   # Additive STDP
                                            "mu_minus": 0.0,  # Additive STDP
                                            "Wmax": -1.0,
                                            "weight": Init_PCDCN,
                                            "delay": 1.0})
    PCDCN_conn_param = {"model": "stdp_synapse"} 
else:
    PCDCN_conn_param = {"model": "static_synapse",
                        "weight": Init_PCDCN,
                        "delay": 1.0}
count_DCN = 0
for P in range(PC_num):
    nest.Connect([PC[P]], [DCN[count_DCN]],
                        'one_to_one', PCDCN_conn_param)
    if PLAST2:
        nest.Connect([PC[P]], [vt2[count_DCN]], 'one_to_one',
                            {"model":  "static_synapse",
                            "weight": 1.0,
                            "delay":  1.0})
    if P % 2 == 1:
        count_DCN += 1
        
        
# Input_generation = nest.Create("spike_generator", MF_num)
# nest.Connect(Input_generation,MF,'one_to_one')
# MFinput_file = open("/home/mizzou/.opt/nrpStorage/USER_DATA/MF_100Trial_VOR.dat",'r')
# for MFi in Input_generation:
#     Spikes_s = MFinput_file.readline()
#     Spikes_s = Spikes_s.split()
#     Spikes_f = []
#     for elements in Spikes_s:
#         Spikes_f.append(float(elements))
#     nest.SetStatus([MFi],{'spike_times' : Spikes_f})
conn1 = nest.GetConnections(source=GR, target=PC)
conn2 = nest.GetConnections(source=MF, target=DCN)
conn3 = nest.GetConnections(source=PC, target=DCN)

population = MF + PC + IO + DCN


circuit = population
