import numpy as np
@nrp.MapVariable("eye_vel", scope=nrp.GLOBAL)
@nrp.MapVariable("V", initial_value=0.0, scope=nrp.GLOBAL)
@nrp.MapVariable("isSaccade", scope=nrp.GLOBAL)
@nrp.MapVariable("idx", initial_value=0)
@nrp.MapVariable("saccade_duration", scope=nrp.GLOBAL)

@nrp.MapSpikeSource("MF_vel", nrp.map_neurons(range(100), lambda i: nrp.brain.mf[i]), nrp.poisson, delay=1.0)
@nrp.Robot2Neuron()
def eye2MF (t, eye_vel, V, isSaccade, MF_vel, idx, saccade_duration):
    def kernel(x):
        y = np.sin(x) * np.exp(-x)
        return y
    y = 3.0 * kernel(np.linspace(0, np.pi, saccade_duration.value))
    if eye_vel.value is not None:
        k = 40
        x = np.ones(100)
        if isSaccade.value == 1:
            input_mf = V.value * y[idx.value] 
            idx.value = idx.value+1
        else:
            input_mf = eye_vel.value
            idx.value = 0
        #clientLogger.info(t, 'input_mf: ', input_mf)
        if input_mf > 0:
            x[50:] = 0
            v = input_mf
            fr_vel = k * v * x
        else:
            x[:50] = 0
            v = -input_mf
            fr_vel = k * v * x
        MF_vel.rate = fr_vel
    else:
        MF_vel.rate = 0.0
        #clientLogger.info('mf: ', MF_pos)
        #clientLogger.info('eyepos: ', eye_position.value)
    
