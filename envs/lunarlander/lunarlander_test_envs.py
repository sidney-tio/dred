from envs.registration import register as gym_register
from .lunarlander_parameterized import LunarLander
#fixed params that were randomized initially
params = [[ -3.58211243,   0.20845414],
          [-10.02311556,   5.33210306],
          [-11.7806391 ,   6.08057095],
          [ -0.67134782,   8.74679624],
          [ -9.98520401,  15.78928294],
          [ -6.74740918,  17.31452842],
          [-11.06398877,  11.04122087],
          [ -9.34399539,  14.07790794],
          [ -2.36235808,   4.30473596],
          [ -4.46353633,  14.59934468]]

class LunarLanderEvalOne(LunarLander):
    def __init__(self):
        gravity = params[0][0]
        wind_power = params[0][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalTwo(LunarLander):
    def __init__(self):
        gravity = params[1][0]
        wind_power = params[1][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalThree(LunarLander):
    def __init__(self):
        gravity = params[2][0]
        wind_power = params[2][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalFour(LunarLander):
    def __init__(self):
        gravity = params[3][0]
        wind_power = params[3][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalFive(LunarLander):
    def __init__(self):
        gravity = params[4][0]
        wind_power = params[4][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalSix(LunarLander):
    def __init__(self):
        gravity = params[5][0]
        wind_power = params[5][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalSeven(LunarLander):
    def __init__(self):
        gravity = params[6][0]
        wind_power = params[6][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalEight(LunarLander):
    def __init__(self):
        gravity = params[7][0]
        wind_power = params[7][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalNine(LunarLander):
    def __init__(self):
        gravity = params[8][0]
        wind_power = params[8][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

class LunarLanderEvalTen(LunarLander):
    def __init__(self):
        gravity = params[9][0]
        wind_power = params[9][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True)
    def reset(self):
        return super()._reset()

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname
gym_register(id = 'LunarLander-Eval-One-v0',
             entry_point=module_path + ':LunarLanderEvalOne',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Two-v0',
             entry_point=module_path + ':LunarLanderEvalTwo',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Three-v0',
             entry_point=module_path + ':LunarLanderEvalThree',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Four-v0',
             entry_point=module_path + ':LunarLanderEvalFour',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Five-v0',
             entry_point=module_path + ':LunarLanderEvalFive',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Six-v0',
             entry_point=module_path + ':LunarLanderEvalSix',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Seven-v0',
             entry_point=module_path + ':LunarLanderEvalSeven',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Eight-v0',
             entry_point=module_path + ':LunarLanderEvalEight',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Nine-v0',
             entry_point=module_path + ':LunarLanderEvalNine',
             max_episode_steps=500)
gym_register(id = 'LunarLander-Eval-Ten-v0',
             entry_point=module_path + ':LunarLanderEvalTen',
             max_episode_steps=500)