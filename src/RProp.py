import tensorflow as tf

class RProp(tf.keras.optimizers.Optimizer):
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., learning_rate=1e-3, **kwargs):
        # Chiama il costruttore della classe base
        super(RProp, self).__init__(learning_rate=learning_rate, **kwargs)

        # Imposta i parametri dell'RProp
        self.init_alpha = init_alpha
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def _create_slots(self, var_list):
        # Crea gli slot per memorizzare i gradienti precedenti e i valori di alpha
        for var in var_list:
            self.add_slot(var, "old_grad", initializer="zeros")  # Slot per i gradienti precedenti
            self.add_slot(var, "alpha", initializer=tf.constant_initializer(self.init_alpha))  # Slot per alpha

    def update_step(self, grad, var, learning_rate=None):
        # Ottieni i gradienti precedenti e i valori di alpha
        old_grad = self.get_slot(var, "old_grad")
        alpha = self.get_slot(var, "alpha")

        # Aggiorna alpha in base al prodotto tra il gradiente corrente e quello precedente
        new_alpha = tf.where(
            grad * old_grad > 0,
            tf.minimum(alpha * self.scale_up, self.max_alpha),
            tf.maximum(alpha * self.scale_down, self.min_alpha)
        )

        # Aggiorna i pesi
        var.assign_sub(tf.sign(grad) * new_alpha)

        # Aggiorna i gradienti precedenti e i valori di alpha
        old_grad.assign(grad)
        alpha.assign(new_alpha)

    def get_config(self):
        config = {
            'init_alpha': self.init_alpha,
            'scale_up': self.scale_up,
            'scale_down': self.scale_down,
            'min_alpha': self.min_alpha,
            'max_alpha': self.max_alpha,
            'learning_rate': self.learning_rate
        }
        base_config = super(RProp, self).get_config()
        return {**base_config, **config}