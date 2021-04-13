@nrp.MapVariable("dt", scope=nrp.GLOBAL)
@nrp.MapVariable("target_freq", scope=nrp.GLOBAL)
@nrp.MapVariable("trialId", initial_value=1)

@nrp.MapCSVRecorder("recorder_weight1", filename="weights1.csv", headers=["time", "trialId", "direction", "weight"])
@nrp.MapCSVRecorder("recorder_weight2", filename="weights2.csv", headers=["time", "trialId", "direction", "weight"])
@nrp.MapCSVRecorder("recorder_weight3", filename="weights3.csv", headers=["time", "trialId", "direction", "weight"])

@nrp.Robot2Neuron()
def record_weights (t, dt, target_freq, trialId, recorder_weight1, recorder_weight2, recorder_weight3):
    T = target_freq.value
    
    #log the first timestep (20ms), each couple of seconds
    if t % (T) < dt.value:
        import nest
        weight1 = nest.GetStatus(nrp.config.brain_root.conn1, keys="weight")
        weight2 = nest.GetStatus(nrp.config.brain_root.conn2, keys="weight")
        weight3 = nest.GetStatus(nrp.config.brain_root.conn3, keys="weight")
        recorder_weight1.record_entry(t, trialId.value, "CP", weight1)
        recorder_weight2.record_entry(t, trialId.value, "CP", weight2)
        recorder_weight3.record_entry(t, trialId.value, "CP", weight3)
    elif t % (T/2) < dt.value:
        import nest
        weight1 = nest.GetStatus(nrp.config.brain_root.conn1, keys="weight")
        weight2 = nest.GetStatus(nrp.config.brain_root.conn2, keys="weight")
        weight3 = nest.GetStatus(nrp.config.brain_root.conn3, keys="weight")
        recorder_weight1.record_entry(t, trialId.value, "CF", weight1)
        recorder_weight2.record_entry(t, trialId.value, "CF", weight2)
        recorder_weight3.record_entry(t, trialId.value, "CF", weight3)
        trialId.value = trialId.value + 1