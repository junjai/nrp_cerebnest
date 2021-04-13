from hbp_nrp_excontrol.logs import clientLogger
@nrp.MapVariable("eye_position", scope=nrp.GLOBAL)
@nrp.MapSpikeSink("positive_dcn", nrp.brain.dcn_p, nrp.leaky_integrator_alpha, delay=1.0)
@nrp.MapSpikeSink("negative_dcn", nrp.brain.dcn_n, nrp.leaky_integrator_alpha, delay=1.0)
@nrp.Neuron2Robot(Topic('/icub/eye_version/pos', std_msgs.msg.Float64))
def eye_control(t, eye_position, positive_dcn, negative_dcn):
    if eye_position.value is None:
        return 0.0
    command = positive_dcn.voltage - negative_dcn.voltage 
    clientLogger.info("Pos: " + str(positive_dcn.voltage) + \
                     " Neg:" + str(negative_dcn.voltage))
    gain = 0.9
    ret = gain * command
    return ret

