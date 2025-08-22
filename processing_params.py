# blur
KSIZE = 3
KSIZE_2 = 3

B_ALPHA = 0.3

# weight
ALPHA = 2
BETA = -0.5

# clahe
CLIP = 2.5
GRID  = (3, 3)

# denoise
H_DENOISE = 0.0001
TMP_WIN_SIZE = 1
SEARCH_WIN_SIZE = 1

base_configs_blue = [
    {"clip": 2.5, "grid": (4, 4)},
    {"clip": 4,   "grid": (6, 6)},
    {"clip": 6,   "grid": (8, 8)},
    {"clip": 8,   "grid": (4, 4)},
    {"clip": 10,  "grid": (8, 8)},
]

base_configs_silver = [
    {"clip": 2.5, "grid": (4, 4)},
    {"clip": 4,   "grid": (6, 6)},
    {"clip": 6,   "grid": (8, 8)},
]

k_sizes = [5, 7, 9]
k_sizes_2 = [3, 5, 7]
counter = 1
blue_alpha_k = [0.3, 0.5, 1.1, 3.0]
base_params_blue = {}
base_params_silver = {}


# Blue
for k_size in k_sizes:
    for config in base_configs_blue:
        base_params_blue[counter] = {
            "k_size": k_size,
            "clip": config["clip"],
            "grid": config["grid"]
        }
        counter += 1

# Silver
for k_size in k_sizes:
    for config in base_configs_silver:
        for blue_alpha in blue_alpha_k:
            for k_size_2 in k_sizes_2:
                base_params_silver[counter] = {
                    "b_alpha": blue_alpha,
                    "k_size": k_size,
                    "k_size_2": k_size_2,
                    "clip": config["clip"],
                    "grid": config["grid"]
                }
                counter += 1

custom_params = {
    "silver": {
        "default": {}
    },
    "blue": {
        "default": {}
    }
}

custom_params["silver"].update(base_params_silver)
custom_params["blue"].update(base_params_blue)
