import numpy as np
@nrp.MapVariable("error", scope=nrp.GLOBAL)
@nrp.MapVariable("true_error", scope=nrp.GLOBAL)
@nrp.MapVariable("isSaccade", scope=nrp.GLOBAL)
@nrp.MapVariable("dt", scope=nrp.GLOBAL)

@nrp.MapSpikeSource("positive_error", nrp.brain.io_p, nrp.poisson, delay=1.0)
@nrp.MapSpikeSource("negative_error", nrp.brain.io_n, nrp.poisson, delay=1.0)
def error_feedback (t, positive_error, negative_error, error, true_error, isSaccade, dt):
    def sigmoid(x, a, b):
        sig = a / (1 + np.exp(-b*x))
        return sig
    default_fr = 1.0
    a = 2 * default_fr
    b = 150.0
    c = b/2
    if error.value is not None and true_error.value is not None and isSaccade.value is not None and dt.value is not None:
    #    err = error.value[0]
        err = true_error.value
        err_normal = err / 45
        if isSaccade.value == 2:
            if err >= 0.0:
                #clientLogger.info('positive error')
                positive_error.rate = c * err_normal + 1.0
                negative_error.rate = sigmoid(-err_normal, a, b)
            elif err < 0.0:
                #clientLogger.info('negative error')
                positive_error.rate = sigmoid(err_normal, a, b)
                negative_error.rate = -c * err_normal + 1.0
        else:
            positive_error.rate = default_fr
            negative_error.rate = default_fr
        clientLogger.info('isSaccade error: ', isSaccade.value, 'error: ', err_normal, 'error_p:', positive_error.rate, 'error_n:', negative_error.rate)


