# Imported Python Transfer Function
#
@nrp.MapVariable("true_error", initial_value=0.0, scope=nrp.GLOBAL)
@nrp.MapVariable("eye_position", scope=nrp.GLOBAL)
@nrp.MapVariable("target_position", scope=nrp.GLOBAL)
@nrp.MapCSVRecorder("recorder_error", filename="true_error.csv", headers=["time", "error"])

@nrp.Robot2Neuron()
def true_error (t, true_error, eye_position, target_position, recorder_error):
    if eye_position.value is not None and target_position.value is not None:
        true_error.value = -target_position.value - eye_position.value * 45
        recorder_error.record_entry(t, true_error.value)

