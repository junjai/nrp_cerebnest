@nrp.MapVariable("dt", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target_freq", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target_ampl", initial_value=None, scope=nrp.GLOBAL)
def set_params (t, dt, target_freq, target_ampl):
    dt.value = 0.01
    target_freq.value = 1.6
    target_ampl.value = 0.1