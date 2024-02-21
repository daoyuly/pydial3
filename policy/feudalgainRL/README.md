## FeudalGain

This is the implementation to the work **What Does The User Want? Information Gain for Hierarchical Dialogue Policy Optimisation**, published at ASRU2021.

Reference: https://arxiv.org/pdf/2109.07129.pdf

#### Train a FeudalGain policy

First of all, create a virtual environment with python3 and run **pip install -r requirements** to install the python packages.

You can find config-files for all environments in the folder pydial3-public/policy/feudalgainRL/configs. To start a training, choose one of the config files and run the following command in the main repo:

```
python3 pydial.py train config_path/config.cfg
```

You can change parameter settings in the config files as needed. The most important parameters to set are:

```
[policy]
noisy_acer=True/False: Use noisy networks for policy \pi_mg or \pi_m and \pi_g
use_pass=True/False: Train information policy \pi_i with transitions where action=pass(). Deactivated if information gain is used. Should be activated if Feudal is used for training.

[feudalpolicy]
only_master = True/False: True means that we use only policy pi_mg, set to False if you want to use pi_m and pi_g
js_threshold = 0.2: threshold for information gain reward calculated by JS-divergence. If set to 1.0, we do use external reward for training \pi_i.

[dqnpolicy]
architecture = noisy_duel/duel: use noisy_duel for noisy network architecture
```

If you want to use the vanilla Feudal algorithm, set the parameters in the config as follows:

```
[policy]
noisy_acer=False
use_pass=True

[feudalpolicy]
only_master = False
js_threshold = 1.0

[dqnpolicy]
architecture = duel
```

Log files and policies will be saved in the directories specified in the config file in section **[exec_config]**.
