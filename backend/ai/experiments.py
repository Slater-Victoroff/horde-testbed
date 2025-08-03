EXPERIMENTS = []
# EXPERIMENTS += [
#     {
#         "name": "L1-campfire",
#         "dataset": "VFX/stylized_flame/campfire01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "L1-fireball",
#         "dataset": "VFX/stylized_flame/fireball01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "L1-flame",
#         "dataset": "VFX/stylized_flame/flame01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

EXPERIMENTS += [
    {
        "name": "fire-circle",
        "dataset": "VFX/fire-circle/",
        "config": {
            "latent_dim": 0,
            "trunk_pos_channels": 2,
            "trunk_time_channels": 1,
            "film_pos_channels": 16,
            "film_time_channels": 8,
            "film_pos_scheme": "spiral",
            "film_time_scheme": "spiral",
            "film_pos_include_raw": False,
            "film_time_include_raw": False,
            "prefilm_dims": 32,
            "hidden_dim": 32,
            "apply_film": [1]
        }
    },
]

# EXPERIMENTS += [
#     {
#         "name": "ossim-small-L1-fireball",
#         "dataset": "VFX/stylized_flame/fireball01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "ossim-small-L1-flame",
#         "dataset": "VFX/stylized_flame/flame01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1]
#         }
#     },
# ]

#     {
#         "name": "spiral+raw-trunk-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 8,
#             "trunk_pos_scheme": "sinusoidal",
#             "trunk_pos_include_raw": True,
#             "trunk_time_channels": 4,
#             "trunk_time_scheme": "sinusoidal",
#             "trunk_time_include_raw": True,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-raw-encoded-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-noproj-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-multifilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1, 2, 3]
#         }
#     },
#     {
#         "name": "spiral-singleseed1-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-singleseed2-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "spiral-pureraw-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 2,
#             "film_time_channels": 1,
#             "film_pos_scheme": None,
#             "film_time_scheme": None,
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-densefilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-deepfilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1, 2]
#         }
#     },
#     {
#         "name": "spiral-delayedfilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [2]
#         }
#     },
#     {
#         "name": "spiral-prefilm16-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 16,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-prefilm64-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-trunknormonly-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 2,
#             "trunk_pos_scheme": None,
#             "trunk_pos_include_raw": False,
#             "trunk_pos_include_norm": True,
#             "trunk_time_channels": 1,
#             "trunk_time_scheme": None,
#             "trunk_time_include_raw": False,
#             "trunk_time_include_norm": True,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-nonorm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "film_pos_include_norm": False,
#             "film_time_include_norm": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-singleseed3-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     }
# ]

# for seed in range(1, 4):
#     EXPERIMENTS += [
#         {
#             "name": f"spiral-shadersmall-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 48,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-shaderdenseproj-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 32,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 16,  # narrow projection
#                 "hidden_dim": 32,
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-shaderwideencproj64-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 32,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 32,
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-gaussianfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "gaussian",
#                 "film_time_scheme": "gaussian",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-linearfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "linear",
#                 "film_time_scheme": "linear",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-polyfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "polynomial",
#                 "film_time_scheme": "polynomial",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#     ]

# EXPERIMENTS += [
#     {
#         "name": "spiral-noproj-wideinput-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "sinus-proj64-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-smallraw-noproj-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 4,
#             "film_time_channels": 2,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [{
#         "name": "spiral-densefilm-5mmnist",
#         "dataset": "moving_mnist",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1],
#             "output_channels": 1,
#         }
#     },
# ]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "sloss-drill-fiber-cloud",
#     "dataset": "VFX/clouds/cloud01",
#     "config": {},
# }]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "soapdrill-fiber-explosion",
#     "dataset": "VFX/explosions/explosion00",
#     "config": {},
# }]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "soapdrill-fiber-flame",
#     "dataset": "VFX/hollow-flame",
#     "config": {},
# }]
