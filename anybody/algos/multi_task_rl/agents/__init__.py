# from .base import MyAgent
from .ppo import MultiEnvPPO
from .ppo_rnn import MultiEnvPPO_RNN
from .sac import MultiEnvSAC
from .td3 import MultiEnvTD3

agents = {
    "ppo": MultiEnvPPO,
    "ppo_rnn": MultiEnvPPO_RNN,
    "sac": MultiEnvSAC,
    "td3": MultiEnvTD3,
    "random": MultiEnvPPO,
}