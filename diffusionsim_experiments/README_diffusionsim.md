# Running diffusion sim code

## Requirements

See requirements.txt file for full pip requirements.\
Major requirements: jax, jaxlib

## Reproducing results in paper

``mkdir diff_data``\
``cp plot_curves.py diff_data/``\
``python diffusionsimsimple.py --keep_frac 1.0 --filename 1.0 --num_opt 800``\
``python diffusionsimsimple.py --keep_frac 0.1 --filename 0.1 --num_opt 800``\
``python diffusionsimsimple.py --keep_frac 0.01 --filename 0.01 --num_opt 800``\
``python diffusionsimsimple.py --keep_frac 0.005 --filename 0.005 --num_opt 800``\
``python diffusionsimsimple.py --keep_frac 0.002 --filename 0.002 --num_opt 800``\
``python diffusionsimsimple.py --keep_frac 0.001 --filename 0.001 --num_opt 800``\
``cd diff_data``\
``python plot_curves.py``
