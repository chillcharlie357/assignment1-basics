from math import cos, pi


def lr_scheduler(cur_iter: int, lr_max: float, lr_min: float, warm_up: int, iterations: int):
    if cur_iter < warm_up:
        return (cur_iter / warm_up) * lr_max
    elif cur_iter <= iterations:
        theta = pi * (cur_iter - warm_up) / (iterations - warm_up)
        return lr_min + 0.5 * (1 + cos(theta)) * (lr_max - lr_min)
    else:
        return lr_min
