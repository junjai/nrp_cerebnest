# CEREBELLUM
PLAST1 = True   # PF-PC ex
PLAST2 = False  # MF-DCN ex
PLAST3 = False  #
 
LTP1 = 0.1
LTD1 = -1.0
 
LTP2 = 1e-5
LTD2 = -1e-6
LTP3 = 1e-7
LTD3 = 1e-6

    if PLAST2:
        vt2 = nest.Create("volume_transmitter_alberto", DCN_num)
        for n, vti in enumerate(vt2):
            nest.SetStatus([vti], {"vt_num": n})
    if PLAST3:
        nest.SetDefaults('stdp_synapse', {"tau_plus": 30.0,
                                                "lambda": LTP3,
                                              "alpha": LTD3/LTP3,
                                              "mu_plus": 0.0,   # Additive STDP
                                              "mu_minus": 0.0,  # Additive STDP
                                              "Wmax": -0.5,
                                              "weight": Init_PCDCN,
                                              "delay": 1.0})
        PCDCN_conn_param = {"model": "stdp_synapse"}
    else:
        PCDCN_conn_param = {"model": "static_synapse",
                            "weight": Init_PCDCN,
                            "delay": 1.0}
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
            nest.SetStatus(A, {'vt_num': float(i)})
    else:
        MFDCN_conn_param = {"model":  "static_synapse",
                            "weight": Init_MFDCN,
                            "delay":  10.0}
        nest.Connect(MF, DCN, 'all_to_all', MFDCN_conn_param)
    # PC-DCN inhibitory plastic connections
    # each DCN receives 2 connections from 2 contiguous PC
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