import tensorflow as tf


def mse_only_wrapper(core):
    def loss_fn(y_true, y_pred):
        return core.mse_only(y_true, y_pred)

    loss_fn.__name__ = "mse_only"
    return loss_fn


def total_loss_wrapper(core, sb_ece, xi_mom):
    def loss_fn(y_true, y_pred):
        total_loss, _, _, _ = core.loss_total(y_true, y_pred, sb_ece=sb_ece, xi_mom=xi_mom)
        return total_loss

    loss_fn.__name__ = "total_loss"
    return loss_fn


def mse_metric_wrapper(core, sb_ece=None, xi_mom=None):
    def metric_fn(y_true, y_pred):
        # if you don’t have loss_total on core, compute directly here
        mean = y_pred[..., 0]
        return tf.reduce_mean(tf.square(y_true - mean))

    metric_fn.__name__ = "mse"
    return metric_fn


def sigma_loss_metric_wrapper(core, sb_ece):
    def metric_fn(y_true, y_pred):
        return sb_ece.compute(y_true, y_pred)

    metric_fn.__name__ = "sb_ece"
    return metric_fn


def xi_mom_loss_metric_wrapper(core, xi_mom):
    def metric_fn(y_true, y_pred):
        return xi_mom.compute(y_true, y_pred)

    metric_fn.__name__ = "xi_mom"
    return metric_fn
