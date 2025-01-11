import tensorflow as tf
import numpy as np

class RProp(tf.keras.optimizers.Optimizer):
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., learning_rate=1e-3, **kwargs):
        # Chiama il costruttore della classe base, passando il parametro 'learning_rate'
        super(RProp, self).__init__(**kwargs)

        # Imposta i parametri dell'RProp
        self.init_alpha = init_alpha
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # Usa il parametro 'learning_rate' della classe base
        self.learning_rate = learning_rate

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        alphas = [tf.Variable(np.ones_like(p) * self.init_alpha) for p in params]
        old_grads = [tf.Variable(np.zeros_like(p)) for p in params]
        self.weights = alphas + old_grads
        self.updates = []

        for param, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            new_alpha = tf.where(
                grad * old_grad > 0,
                tf.minimum(alpha * self.scale_up, self.max_alpha),
                tf.maximum(alpha * self.scale_down, self.min_alpha)
            )
            new_param = param - tf.sign(grad) * new_alpha
            # Apply constraints
            if param in constraints:
                c = constraints[param]
                new_param = c(new_param)
            self.updates.append(param.assign(new_param))
            self.updates.append(alpha.assign(new_alpha))
            self.updates.append(old_grad.assign(grad))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': self.init_alpha,
            'scale_up': self.scale_up,
            'scale_down': self.scale_down,
            'min_alpha': self.min_alpha,
            'max_alpha': self.max_alpha,
            'learning_rate': self.learning_rate  # Conserve 'learning_rate' nella configurazione
        }
        base_config = super(RProp, self).get_config()
        return {**base_config, **config}
