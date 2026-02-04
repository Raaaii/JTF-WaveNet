import tensorflow as tf


def _set_layer_tree_trainable(layer_or_obj, trainable: bool):
    """
    Sets .trainable on a Layer or recursively on containers (lists/dicts).
    """
    if layer_or_obj is None:
        return

    if isinstance(layer_or_obj, tf.keras.layers.Layer):
        layer_or_obj.trainable = trainable
        return

    if isinstance(layer_or_obj, dict):
        for v in layer_or_obj.values():
            _set_layer_tree_trainable(v, trainable)
        return

    if isinstance(layer_or_obj, (list, tuple)):
        for v in layer_or_obj:
            _set_layer_tree_trainable(v, trainable)
        return


def _set_scalar_weight_trainable(model, weight_attr: str, trainable: bool):
    """
    Handles tf.Variable scalars that Keras tracks as weights.
    Keras decides trainable variables from model.trainable_weights,
    which is backed by model._trainable_weights / model._non_trainable_weights.
    """
    if not hasattr(model, weight_attr):
        return

    w = getattr(model, weight_attr)
    if not isinstance(w, tf.Variable):
        return

    # Move between internal lists
    tw = list(getattr(model, "_trainable_weights", []))
    ntw = list(getattr(model, "_non_trainable_weights", []))

    if trainable:
        if w in ntw:
            ntw.remove(w)
        if w not in tw:
            tw.append(w)
    else:
        if w in tw:
            tw.remove(w)
        if w not in ntw:
            ntw.append(w)

    model._trainable_weights = tw
    model._non_trainable_weights = ntw


def set_mean_branch_trainable(model, trainable: bool):
    # mean expansion
    _set_layer_tree_trainable(getattr(model, "x_dense", None), trainable)

    # mean wavenet blocks
    _set_layer_tree_trainable(getattr(model, "mean_time_layers", None), trainable)
    _set_layer_tree_trainable(getattr(model, "mean_freq_layers", None), trainable)

    # mean trunk + head
    _set_layer_tree_trainable(getattr(model, "mean_final_conv", None), trainable)
    _set_layer_tree_trainable(getattr(model, "mean_shared_trunk", None), trainable)
    _set_layer_tree_trainable(getattr(model, "mean_head_conv", None), trainable)
    _set_layer_tree_trainable(getattr(model, "mean_head_dense", None), trainable)

    # optional dropout
    if getattr(model, "use_dropout", False):
        _set_layer_tree_trainable(getattr(model, "mean_final_dropout", None), trainable)


def set_error_branch_trainable(model, trainable: bool, train_error_scalars: bool = True):
    # error expansion
    _set_layer_tree_trainable(getattr(model, "x_dense_error", None), trainable)

    # error wavenet blocks
    _set_layer_tree_trainable(getattr(model, "error_time_layers", None), trainable)
    _set_layer_tree_trainable(getattr(model, "error_freq_layers", None), trainable)

    # error trunk + head
    _set_layer_tree_trainable(getattr(model, "error_final_conv", None), trainable)
    _set_layer_tree_trainable(getattr(model, "error_shared_trunk", None), trainable)
    _set_layer_tree_trainable(getattr(model, "error_head", None), trainable)
    _set_layer_tree_trainable(getattr(model, "error_head_dense", None), trainable)

    # optional dropout
    if getattr(model, "use_dropout", False):
        _set_layer_tree_trainable(getattr(model, "error_final_dropout", None), trainable)

    # IMPORTANT: scalar tf.Variables (not layers)
    if train_error_scalars:
        _set_scalar_weight_trainable(model, "error_offset", trainable)
        _set_scalar_weight_trainable(model, "error_softplus_log_beta", trainable)
    else:
        _set_scalar_weight_trainable(model, "error_offset", False)
        _set_scalar_weight_trainable(model, "error_softplus_log_beta", False)
