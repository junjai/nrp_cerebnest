# Imported Python Transfer Function
#
from std_msgs.msg import Float64
@nrp.MapVariable("dt", scope=nrp.GLOBAL)
@nrp.MapVariable("error", scope=nrp.GLOBAL)
@nrp.MapVariable("true_error", scope=nrp.GLOBAL)
@nrp.MapVariable("eye_position", scope=nrp.GLOBAL)
@nrp.MapVariable("eye_vel", scope=nrp.GLOBAL)
@nrp.MapVariable("isSaccade", initial_value=0, scope=nrp.GLOBAL)
@nrp.MapVariable("X", initial_value=0.0)
@nrp.MapVariable("V", initial_value=0.0, scope=nrp.GLOBAL)
@nrp.MapVariable("idx", initial_value=0)
@nrp.MapVariable("idx_error", initial_value=0, scope=nrp.GLOBAL)
@nrp.MapCSVRecorder("eye_state", filename="eye_state.csv", headers=["time", "state"])
@nrp.MapVariable("DCN_output", scope=nrp.GLOBAL)
@nrp.MapVariable("saccade_duration", initial_value=0, scope=nrp.GLOBAL)

#@nrp.MapSpikeSink("result_0_dv", nrp.brain.actors[1], nrp.leaky_integrator_alpha)
#@nrp.MapSpikeSink("result_1_dv", nrp.brain.actors[2], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/icub/eye_version/vel', Float64))
def move_eye (t, eye_position, eye_vel, error, true_error, X, V, idx, idx_error, isSaccade, dt, eye_state, DCN_output, saccade_duration):
    if dt.value is not None and error.value is not None and true_error.value is not None:
        def deg2rad(deg):
            """
            Degrees to radians conversion function.
            :param deg: value in degrees
            :return: value of deg in radians
            """
            return (float(deg) / 360.) * (2. * np.pi)
        if eye_position.value is None:
            return 0.0
        if eye_vel.value is None:
            return 0.0
        maxAngle = 45.0
        min_error = 5.0/maxAngle
        k = 20.0
        T = 2.0/k
        L = T // dt.value
        tt = np.arange(L) / L
        saccade_duration.value = L

        T_error = 0.25
        L_error = T_error // dt.value
        tt_error = np.arange(L_error) / L_error
        clientLogger.info(t, 'error:', error.value[0], 'true error', true_error.value)

# using error from camera image or from discrepancy between eye and target position 
        # err = error.value[0]
        err = true_error.value

        if isSaccade.value == 0: # default phase
            #clientLogger.info('not saccade: ', isSaccade.value)
            #clientLogger.info('V: ', V.value, 'T:', T, 'X: ', X.value)
            ret = 0
            if abs(err/maxAngle) > min_error:
                isSaccade.value = 1
                X.value = err / maxAngle
                V.value = k * X.value
                idx.value = 0
                clientLogger.info('X:', X.value, 'V:', V.value)
        elif isSaccade.value == 1: # saccade phase
            #clientLogger.info('is saccade: ', tt)
            vel = V.value * (-np.cos(2 * np.pi * tt[idx.value]) + 1) 
            ret = vel + DCN_output.value
            #clientLogger.info('V:', V.value)
            clientLogger.info(t, 'ret: ', ret, 'idx: ', idx.value, 'X:', X.value, 'error:', error.value[0]/maxAngle)
            idx.value = idx.value + 1
            
            if idx.value >= len(tt):
                isSaccade.value = 2
        elif isSaccade.value == 2: # error feedback phase
            idx_error.value = idx_error.value + 1
            
            if idx_error.value >= len(tt_error):
                isSaccade.value = 0
                idx_error.value = 0
            ret = 0   
            #ret = DCN_output.value
        else:
            isSaccade.value = 0
            ret = 0
        if isSaccade.value == 0:
            ret = 0.5 * err 
        
        eye_state.record_entry(t, isSaccade.value)
        clientLogger.info(t, 'state:', isSaccade.value, 'vel: ', ret, 'pos:', eye_position.value)
        return ret
    #

