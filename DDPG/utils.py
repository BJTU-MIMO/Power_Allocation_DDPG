import numpy as np
import tensorflow as tf
import pdb

def huber_loss(x, delta=1.):
    """
    Compute the huber loss.
    https://en.wikipedia.org/wiki/Huber_loss

    :param x (np.ndarray or tf.Tensor): Values to compute the huber loss.
    :param delta (float): Positive floating point value. Represents the
                          maximum possible gradient magnitude.
    :return (tf.Tensor): The huber loss.
    """
    delta = tf.ones_like(x) * delta
    less_than_max = 0.5 * tf.square(x)
    greater_than_max = delta * (tf.abs(x) - 0.5 * delta)
    return tf.where(
        tf.abs(x) <= delta,
        x=less_than_max,
        y=greater_than_max)

def normalize(ot):
    return ot

def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):
    """
    Returns an op to update a list of target variables from source variables.

    The update rule is:
    `target_variable = (1 - tau) * target_variable + tau * source_variable`.

    :param target_variables: a list of the variables to be updated.
    :param source_variables: a list of the variables used for the update.
    :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
        with small tau representing an incremental update, and tau == 1
        representing a full update (that is, a straight copy).
    :param use_locking: use `tf.Variable.assign`'s locking option when assigning
        source variable values to target variables.
    :param name: sets the `name_scope` for this op.
    :raise TypeError: when tau is not a Python float
    :raise ValueError: when tau is out of range, or the source and target variables
        have different numbers or shapes.
    :return: An op that executes all the variable updates.
    """
    if not isinstance(tau, float):
        raise TypeError("Tau has wrong type (should be float) {}".format(tau))
    if not 0.0 < tau <= 1.0:
        raise ValueError("Invalid parameter tau {}".format(tau))
    if len(target_variables) != len(source_variables):
        raise ValueError("Number of target variables {} is not the same as "
                         "number of source variables {}".format(
                             len(target_variables), len(source_variables)))

    same_shape = all(trg.get_shape() == src.get_shape()
                     for trg, src in zip(target_variables, source_variables))
    if not same_shape:
        raise ValueError("Target variables don't have the same shape as source "
                         "variables.")

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, use_locking)

    # with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)