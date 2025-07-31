# blur
KSIZE = (5, 5)

# weight
ALPHA = 2
BETA = -0.5

# clahe
CLIP = 2.5
GRID  = (8, 8)

# denoise
H_DENOISE = 0.0001
TMP_WIN_SIZE = 1
SEARCH_WIN_SIZE = 1

custom_params = {
    "default": {},
    1: {
        "clip": 10,
    },
    2: {
        "k_size": (7, 7),
        "alpha": 3,
        "clip": 5,
    },
    3: {
        "clip": 10,
        "grid": (4, 4),
    },
    4: {
        "k_size": (3, 3),
        "alpha": 6,
        "grid": (6, 6)
    },
    5: {
        "clip": 10,
        "grid": (2, 2)
    },
    6: {
        "beta": -1.5,
        "clip": 12,
    },
    7: {
        "k_size": (3, 3),
        "alpha": 3,
        "beta": -1.5,
        "clip": 12,
        "grid": (2, 2)
    },
    8: {
        "clip": 3,
        "alpha": 3,
        "grid": (2, 2),
        "h_denoise": 2,
        "tmp_win_size": 4,
        "search_win_size": 20,
    }
}