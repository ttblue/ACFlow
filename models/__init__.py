
def get_model(hps):
    if hps.model == 'flow':
        from .flow import Model
        model = Model(hps)
    elif hps.model == 'flow_gan':
        from .flow_gan import Model
        model = Model(hps)
    elif hps.model == 'tan':
        from .tan import Model
        model = Model(hps)
    elif hps.model == 'cond_tan':
        from .cond_tan import Model
        model = Model(hps)
    else:
        raise Exception()

    return model
