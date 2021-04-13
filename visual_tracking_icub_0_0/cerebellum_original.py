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
LTP1 = 0.06 #0.05 #0.01
LTD1 = -0.9 #-0.4 #-0.5
Init_PFPC = 4.0

nest.SetKernelStatus({'resolution' : 1.0})

def create_brain():
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

    PLAST = True
    
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
    
    if PLAST:
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
    Init_MFDCN = 0.07
    MFDCN_conn_param = {"model":  "static_synapse",
                        "weight": Init_MFDCN,
                        "delay":  10.0}
    nest.Connect(MF, DCN, 'all_to_all', MFDCN_conn_param)

    
    # PC-DCN inhibitory plastic connections
    # each DCN receives 2 connections from 2 contiguous PC
    Init_PCDCN = -0.5 #-0.5 
    PCDCN_conn_param = {"model": "static_synapse",
                        "weight": Init_PCDCN,
                        "delay": 1.0}
    count_DCN = 0
    for P in range(PC_num):
        nest.Connect([PC[P]], [DCN[count_DCN]],
                     'one_to_one', PCDCN_conn_param)
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
    
    population = MF + PC + IO + DCN

    return population


circuit = create_brain()
