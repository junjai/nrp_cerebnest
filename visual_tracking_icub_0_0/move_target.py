# Imported Python Transfer Function
#
from gazebo_msgs.srv import SetModelState
import rospy
import numpy as np
rospy.wait_for_service("/gazebo/set_model_state")
service_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=True)
@nrp.MapVariable("dt", scope=nrp.GLOBAL)
@nrp.MapVariable("target_freq", scope=nrp.GLOBAL)
@nrp.MapVariable("target_ampl", scope=nrp.GLOBAL)
@nrp.MapVariable("target_center", initial_value={'x': 0, 'y': 0.4, 'z': 1.0})
@nrp.MapVariable("target_position",initial_value=0.0, scope=nrp.GLOBAL)
@nrp.MapCSVRecorder("trial_clock", filename="clock.csv", headers=["time", "trial"])
@nrp.MapCSVRecorder("recorder_target", filename="target_position.csv", headers=["time", "x"])
@nrp.MapVariable("set_model_state_srv", initial_value=service_proxy)
@nrp.Robot2Neuron() # dummy R2N
def move_target (t, dt, target_freq, target_ampl, target_center, target_position, trial_clock, recorder_target, set_model_state_srv):
    if dt.value is not None and target_freq.value is not None and target_ampl.value is not None:
        ms_msg =  gazebo_msgs.msg.ModelState()
        #frequency = target_freq.value
        T = target_freq.value
        amplitude = target_ampl.value
        center = target_center.value
        ms_msg.model_name = 'Target'
        # set orientation RYP axes
        ms_msg.pose.orientation.x = 0
        ms_msg.pose.orientation.y = 1
        ms_msg.pose.orientation.z = 1
        # reference frame
        ms_msg.reference_frame = 'world'
        #pose
        # ms_msg.pose.position.x = \
            # center['x'] + np.sin(t * frequency * 2 * np.pi) * (float(amplitude) / 2)
        if np.sin(t * 2 * np.pi / T) < 0:
            ms_msg.pose.position.x = center['x'] + float(amplitude) 
        else:
            ms_msg.pose.position.x = center['x']
        
        ms_msg.pose.position.y = center['y']
        ms_msg.pose.position.z = center['z']
        #scale
        ms_msg.scale.x = ms_msg.scale.y = ms_msg.scale.z = 1.0
        #call service
        response = set_model_state_srv.value(ms_msg)
        #check response
        if not response.success:
            clientLogger.info(response.status_message)

        # record clock
        if t % (T) < dt.value:
            trial_clock.record_entry(t, 1)
        else:
            trial_clock.record_entry(t, 0)
        recorder_target.record_entry(t, ms_msg.pose.position.x)
        target_position.value = np.rad2deg(np.arctan(ms_msg.pose.position.x / center['y'])) 
        # clientLogger.info('target pos: ', ms_msg.pose.position.x, target_position.value)
        
#

