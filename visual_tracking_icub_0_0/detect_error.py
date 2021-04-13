# Imported Python Transfer Function
#
@nrp.MapRobotSubscriber("camera_l", Topic('/icub/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("camera_r", Topic('/icub/icub_model/right_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapVariable("error", initial_value=0.0, scope=nrp.GLOBAL)
@nrp.MapCSVRecorder("recorder_error", filename="error.csv", headers=["time", "error", "error_l", "error_r"])
#@nrp.MapSpikeSource("red_left_eye", nrp.brain.sensors[slice(0, 3, 2)], nrp.poisson)
#@nrp.MapSpikeSource("red_right_eye", nrp.brain.sensors[slice(1, 4, 2)], nrp.poisson)
#@nrp.MapSpikeSource("green_blue_eye", nrp.brain.sensors[4], nrp.poisson)
@nrp.Robot2Neuron()
def detect_error (t, camera_l, camera_r, error, recorder_error):
    #import math
    tf = hbp_nrp_cle.tf_framework.tf_lib
    xy_ball_pos_l = tf.find_centroid_hsv(camera_l.value, [50, 100, 100], [70, 255, 255]) or (160, 120)
    xy_ball_pos_r = tf.find_centroid_hsv(camera_r.value, [50, 100, 100], [70, 255, 255]) or (160, 120)

    ae_ball_pos_l = tf.cam.pixel2angle(xy_ball_pos_l[0], xy_ball_pos_l[1])
    ae_ball_pos_r = tf.cam.pixel2angle(xy_ball_pos_r[0], xy_ball_pos_r[1])
    ae_ball_pos = [(ae_ball_pos_l[0]+ae_ball_pos_r[0])/2, (ae_ball_pos_l[1]+ae_ball_pos_r[1])/2]
    error.value = ae_ball_pos # error in degree (a, e)
    recorder_error.record_entry(t, ae_ball_pos[0], ae_ball_pos_l[0], ae_ball_pos_r[0])
    clientLogger.info(t, 'error: ', error.value, 'left: ', ae_ball_pos_l, 'right: ', ae_ball_pos_r)
#    red = 76800.0 / (1.0 + math.exp(-ae_ball_pos[0]))
#    red_left_eye.rate = 1000.0 * red / 76800.0
#    red_right_eye.rate = 1000.0 * red / 76800.0
#    green_blue_eye.rate = 1000.0 * (76800.0 - red) / 76800.0
#

