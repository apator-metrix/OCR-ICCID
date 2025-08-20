# blur
KSIZE = 3

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

base_configs = [
    {"clip": 2.5, "grid": (4, 4)},
    {"clip": 4,   "grid": (6, 6)},
    {"clip": 6,   "grid": (8, 8)},
    {"clip": 8,   "grid": (4, 4)},
    {"clip": 10,  "grid": (8, 8)},
]

k_sizes = [5, 7, 9]
counter = 1
base_params = {}

for k_size in k_sizes:
    for config in base_configs:
        base_params[counter] = {
            "k_size": k_size,
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

# custom_params["silver"].update(base_params)
custom_params["blue"].update(base_params)
