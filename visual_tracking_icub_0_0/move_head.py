import numpy as np
@nrp.MapRobotPublisher('head_rotation', Topic('/icub/neck_yaw/pos', std_msgs.msg.Float64))
@nrp.MapCSVRecorder("recorder_neck", filename="neck_angle.csv", headers=["time", "Neck Angle"])
def move_head (t, head_rotation, recorder_neck):
    HR = 30.0 * (np.pi / 180.0) * np.sin(2. * np.pi * t)
    head_rotation.send_message(std_msgs.msg.Float64(HR))
    recorder_neck.record_entry(t, HR)
