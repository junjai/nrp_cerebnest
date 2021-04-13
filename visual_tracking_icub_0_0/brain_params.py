@nrp.MapVariable("dt", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target_freq", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target_ampl", initial_value=None, scope=nrp.GLOBAL)

@nrp.MapCSVRecorder("recorder_brain", filename="brain_parameters.csv", headers=["dt", "frequency", "amplitude", "LTP1", "LTD1", "LTP2", "LTD2", "LTP3", "LTD3"])

@nrp.Robot2Neuron()
def brain_params (t, recorder_brain, dt, target_freq, target_ampl):
    #log the first timestep (20ms), each couple of seconds
    if t == 0:
        import nest
        LTP1 = nrp.config.brain_root.LTP1
        LTD1 = nrp.config.brain_root.LTD1
        LTP2 = nrp.config.brain_root.LTP2
        LTD2 = nrp.config.brain_root.LTD2
        LTP3 = nrp.config.brain_root.LTP3
        LTD3 = nrp.config.brain_root.LTD3
        recorder_brain.record_entry(dt.value, target_freq.value, target_ampl.value, LTP1, LTD1, LTP2, LTD2, LTP3, LTD3)
        clientLogger.info('Time: ', t)