"""Constant optimizer used for deep symbolic optimization."""

import os
import time
from functools import partial

import numpy as np
from scipy.optimize import minimize

# Optional scipy iteration logging. Enabled by setting DSO_SCIPY_LOG env var
# to a file path. Each worker appends one line per scipy.minimize call.
# Format: pid,n_const,nfev,nit,status,wall_sec,fun
_SCIPY_LOG_PATH = os.environ.get("DSO_SCIPY_LOG")


def make_const_optimizer(name, **kwargs):
    """Returns a ConstOptimizer given a name and keyword arguments"""

    const_optimizers = {
        None: Dummy,
        "dummy": Dummy,
        "scipy": ScipyMinimize,
    }

    return const_optimizers[name](**kwargs)


class ConstOptimizer(object):
    """Base class for constant optimizer"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, f, x0):
        """
        Optimizes an objective function from an initial guess.

        The objective function is the negative of the base reward (reward
        without penalty) used for training. Optimization excludes any penalties
        because they are constant w.r.t. to the constants being optimized.

        Parameters
        ----------
        f : function mapping np.ndarray to float
            Objective function (negative base reward).

        x0 : np.ndarray
            Initial guess for constant placeholders.

        Returns
        -------
        x : np.ndarray
            Vector of optimized constants.
        """
        raise NotImplementedError


class Dummy(ConstOptimizer):
    """Dummy class that selects the initial guess for each constant"""

    def __init__(self, **kwargs):
        super(Dummy, self).__init__(**kwargs)

    def __call__(self, f, x0):
        return x0


class ScipyMinimize(ConstOptimizer):
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        super(ScipyMinimize, self).__init__(**kwargs)

    def __call__(self, f, x0):
        t0 = time.time() if _SCIPY_LOG_PATH else None
        with np.errstate(divide="ignore"):
            opt_result = partial(minimize, **self.kwargs)(f, x0)
        if _SCIPY_LOG_PATH is not None:
            wall = time.time() - t0
            line = "{},{},{},{},{},{:.6f},{:.6g}\n".format(
                os.getpid(),
                len(x0),
                int(opt_result.get("nfev", -1)),
                int(opt_result.get("nit", -1)),
                int(opt_result.get("status", -1)),
                wall,
                float(opt_result.get("fun", float("nan"))),
            )
            # POSIX append is atomic for small writes; safe across pool workers.
            with open(_SCIPY_LOG_PATH, "a") as _f:
                _f.write(line)
        x = opt_result["x"]
        return x
