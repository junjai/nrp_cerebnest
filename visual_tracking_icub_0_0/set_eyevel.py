from sensor_msgs.msg import JointState
@nrp.MapVariable("eye_vel", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("joints", Topic("/icub/joints", JointState))
@nrp.MapCSVRecorder("recorder_eyevel", filename="eye_velocity.csv", headers=["time", "eye velocity"])
@nrp.Robot2Neuron()
def set_eyevel (t, eye_vel, joints, recorder_eyevel):
    joints = joints.value
    if joints is not None:
        eye_vel.value = joints.velocity[joints.name.index('eye_version')]
        recorder_eyevel.record_entry(t, joints.velocity[joints.name.index('eye_version')])
    
        