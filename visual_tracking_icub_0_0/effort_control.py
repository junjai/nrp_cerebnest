import gazebo_msgs.srv
from rospy import ServiceProxy, wait_for_service
from rospy import Duration

wait_for_service('/gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
duration_val = Duration.from_sec(0.001)
@nrp.MapVariable("proxy", initial_value=service_proxy)
@nrp.MapVariable("duration", initial_value=duration_val)
@nrp.Neuron2Robot()
def target_reach_torque(t, proxy, duration):
     proxy.value.call('arm_1_joint', total_torque, None, duration.value)
