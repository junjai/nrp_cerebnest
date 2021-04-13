import numpy as np
@nrp.MapVariable("eye_position", scope=nrp.GLOBAL)
@nrp.MapVariable("indexx", initial_value=0)
@nrp.MapVariable("bufferr", initial_value=np.zeros(5))
@nrp.MapSpikeSink("positive_dcn", nrp.brain.dcn_p, nrp.population_rate)
@nrp.MapSpikeSink("negative_dcn", nrp.brain.dcn_n, nrp.population_rate)
@nrp.Neuron2Robot(Topic('/icub/eye_version/pos', std_msgs.msg.Float64))
def eye_control (t, eye_position, positive_dcn, negative_dcn, indexx, bufferr):
    if eye_position.value is None:
        return 0.0
    command = positive_dcn.rate - negative_dcn.rate 
    bufferr.value[indexx.value] = command
    indexx.value = indexx.value + 1
    if indexx.value == 5:
        indexx.value = 0
    mean_comm = np.mean(bufferr.value)
    gain = 0.009
    ret = gain * mean_comm
    return ret