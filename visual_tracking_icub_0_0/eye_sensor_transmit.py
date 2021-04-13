import numpy as np
@nrp.MapRobotSubscriber("camera", Topic('/icub/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("positive_error", nrp.brain.io_p, nrp.poisson, delay=1.0)
@nrp.MapSpikeSource("negative_error", nrp.brain.io_n, nrp.poisson, delay=1.0)
def eye_sensor_transmit(t, camera, positive_error, negative_error):
    import math
    tf = hbp_nrp_cle.tf_framework.tf_lib
    xy_ball_pos = tf.find_centroid_hsv(camera.value, [50, 100, 100], [70, 255, 255]) \
        or (160, 120)
    ae_ball_pos = tf.cam.pixel2angle(xy_ball_pos[0], xy_ball_pos[1])
    if ae_ball_pos[0] >= 0.0:
        positive_error.rate = 10.0 * np.fabs(ae_ball_pos[0] / 30.0)
        negative_error.rate = 0.01
    elif ae_ball_pos[0] < 0.0:
        negative_error.rate = 10.0 * np.fabs(ae_ball_pos[0] / 30.0)
        positive_error.rate = 0.01
