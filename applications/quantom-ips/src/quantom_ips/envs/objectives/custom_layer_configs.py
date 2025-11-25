def get_layer_config(name):
    if name == "mlp_discriminator":
        return mlp_discriminator
    elif name == "mlp_1d_discriminator":
        return mlp_1d_discriminator
    elif name == "wasserstein_critic":
        return wasserstein_critic
    elif name == "ellipse_discriminator":
        return ellipse_discriminator
    else:
        raise KeyError(f"{name} layer config not found.")


ellipse_discriminator = {
    "Linear_0": {
        "index": 0,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 2, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_1": {
        "index": 1,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_2": {
        "index": 2,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Output": {
        "index": 3,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 1},
        "weight_init": "kaiming_normal",
        "activation": {"class": "Sigmoid"},
    },
}

wasserstein_critic = {
    "Linear_0": {
        "index": 0,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 2, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_1": {
        "index": 1,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_2": {
        "index": 2,
        "bias_init": "zeros",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_3": {
        "index": 3,
        "bias_init": "normal",
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 1},
        "weight_init": "xavier_normal",
    },
}

mlp_1d_discriminator = {
    "Linear_1": {
        "index": 0,
        "class": "Linear",
        "config": {"in_features": 1, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_2": {
        "index": 1,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_3": {
        "index": 2,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_4": {
        "index": 3,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_5": {
        "index": 4,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 1},
        "weight_init": "xavier_normal",
        "bias_init": "normal",
        "activation": {"class": "Sigmoid"},
    },
}

mlp_discriminator = {
    "Linear_1": {
        "index": 0,
        "class": "Linear",
        "config": {"in_features": 2, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_2": {
        "index": 1,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_3": {
        "index": 2,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_4": {
        "index": 3,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 128},
        "weight_init": "kaiming_normal",
        "bias_init": "normal",
        "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
    },
    "Linear_5": {
        "index": 4,
        "class": "Linear",
        "config": {"in_features": 128, "out_features": 1},
        "weight_init": "xavier_normal",
        "bias_init": "normal",
        "activation": {"class": "Sigmoid"},
    },
}
