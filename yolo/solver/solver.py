"""
Abstract class of solvers
"""
class Solver(object):

    def __init__(self, dataset, net, comman_params, solver_params):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError