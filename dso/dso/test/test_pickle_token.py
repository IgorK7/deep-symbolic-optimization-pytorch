"""Pickle round-trip tests for Token and its subclasses.

Covers the multiprocessing-safety patch in library.py: every Token class
must pickle cleanly so that DSO can run with n_cores_batch > 1 (in which
case Programs containing Tokens are sent across mp.Pool workers).

Without the patch, Tokens whose underlying callable has __module__ ==
"__main__" (which can happen via sys.path manipulation or "python -m"
invocation) raise:
    _pickle.PicklingError: Can't pickle <fn>: attribute lookup <name>
                           on __main__ failed
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from dso.functions import function_map
from dso.library import (
    DiscreteAction,
    HardCodedConstant,
    PlaceholderConstant,
    Polynomial,
    StateChecker,
    Token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def round_trip(token):
    """Pickle then unpickle a token; return the restored instance."""
    return pickle.loads(pickle.dumps(token))


def assert_callable_equivalent(orig, restored, sample_inputs):
    """Both tokens must produce the same output on the same inputs."""
    if orig.arity == 0:
        # Constant tokens: function takes no args.
        assert np.allclose(orig.function(), restored.function()), (
            f"Function output differs after pickle round-trip for {orig.name}"
        )
        return
    a = np.asarray(orig.function(*sample_inputs))
    b = np.asarray(restored.function(*sample_inputs))
    assert np.allclose(a, b, equal_nan=True), (
        f"Function output differs after pickle round-trip for {orig.name}"
    )


# ---------------------------------------------------------------------------
# Base Token: unprotected ops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_name",
    ["add", "sub", "mul", "div", "exp", "log", "sqrt", "indicator", "relu", "sigmoid"],
)
def test_unprotected_op_round_trip(op_name):
    canonical = function_map[op_name]
    restored = round_trip(canonical)
    assert restored.name == canonical.name
    assert restored.arity == canonical.arity
    assert restored.complexity == canonical.complexity
    if canonical.arity == 1:
        x = np.array([0.5, 1.0, 2.0, -0.5])
        assert_callable_equivalent(canonical, restored, [x])
    elif canonical.arity == 2:
        x1 = np.array([0.5, 1.0, 2.0])
        x2 = np.array([0.5, 1.0, 0.5])
        assert_callable_equivalent(canonical, restored, [x1, x2])


@pytest.mark.parametrize(
    "op_name",
    [
        "protected_div",
        "protected_exp",
        "protected_log",
        "protected_sqrt",
        "protected_relu",
        "protected_indicator",
    ],
)
def test_protected_op_round_trip(op_name):
    canonical = function_map[op_name]
    restored = round_trip(canonical)
    assert restored.name == canonical.name
    assert restored.arity == canonical.arity
    if canonical.arity == 1:
        x = np.array([0.5, 1.0, 2.0, -0.5])
        assert_callable_equivalent(canonical, restored, [x])
    elif canonical.arity == 2:
        x1 = np.array([0.5, 1.0, 2.0])
        x2 = np.array([0.5, 1.0, 0.5])
        assert_callable_equivalent(canonical, restored, [x1, x2])


# ---------------------------------------------------------------------------
# Indicator specifically: this is the failing case from the cv_rerun grid.
# ---------------------------------------------------------------------------
def test_indicator_token_pickle_targeted():
    """Direct round-trip of the indicator Token. This is the case that
    crashed the cv_rerun grid with n_cores_batch=-1."""
    canonical = function_map["indicator"]
    bytes_ = pickle.dumps(canonical)  # must NOT raise PicklingError
    restored = pickle.loads(bytes_)
    x = np.array([-2.0, -0.5, 0.0, 0.001, 1.0, 100.0])
    expected = np.where(x > 0, 1.0, 0.0)
    assert np.allclose(restored.function(x), expected)


def test_protected_indicator_token_pickle_targeted():
    canonical = function_map["protected_indicator"]
    restored = round_trip(canonical)
    # protected_indicator is numerically equivalent to indicator on bounded x
    x = np.array([-2.0, -0.5, 0.0, 0.001, 1.0, 100.0])
    expected = np.where(x > 0, 1.0, 0.0)
    assert np.allclose(restored.function(x), expected)


# ---------------------------------------------------------------------------
# Input variable Token (function is None)
# ---------------------------------------------------------------------------
def test_input_variable_token():
    tok = Token(function=None, name="x1", arity=0, complexity=1, input_var=0)
    restored = round_trip(tok)
    assert restored.name == "x1"
    assert restored.arity == 0
    assert restored.complexity == 1
    assert restored.input_var == 0
    assert restored.function is None


# ---------------------------------------------------------------------------
# HardCodedConstant
# ---------------------------------------------------------------------------
def test_hard_coded_constant_round_trip():
    tok = HardCodedConstant(value=3.5)
    restored = round_trip(tok)
    assert isinstance(restored, HardCodedConstant)
    assert restored.name == tok.name
    assert np.allclose(restored.value, tok.value)
    # function() returns self.value -- bound method must be re-bound.
    assert np.allclose(restored.function(), tok.function())


def test_hard_coded_constant_with_explicit_name():
    tok = HardCodedConstant(value=2.0, name="two")
    restored = round_trip(tok)
    assert restored.name == "two"
    assert np.allclose(restored.value, tok.value)


# ---------------------------------------------------------------------------
# PlaceholderConstant
# ---------------------------------------------------------------------------
def test_placeholder_constant_unset():
    tok = PlaceholderConstant()
    assert tok.value is None
    restored = round_trip(tok)
    assert isinstance(restored, PlaceholderConstant)
    assert restored.name == "const"
    assert restored.value is None


def test_placeholder_constant_set_after_bfgs():
    tok = PlaceholderConstant(value=1.42)
    restored = round_trip(tok)
    assert isinstance(restored, PlaceholderConstant)
    assert restored.name == "const"
    assert np.allclose(restored.value, tok.value)
    # function() returns self.value via bound method
    assert np.allclose(restored.function(), tok.function())


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------
def test_polynomial_round_trip():
    # p(x1, x2) = 2.0*x1 + 3.0*x2**2
    exponents = [(1, 0), (0, 2)]
    coef = np.array([2.0, 3.0])
    tok = Polynomial(exponents=exponents, coef=coef)
    restored = round_trip(tok)
    assert isinstance(restored, Polynomial)
    assert restored.exponents == exponents
    assert np.allclose(restored.coef, coef)
    # eval_poly should work after round-trip
    X = np.array([[1.0, 2.0], [3.0, 1.0]])
    assert np.allclose(restored.eval_poly(X), tok.eval_poly(X))


# ---------------------------------------------------------------------------
# StateChecker
# ---------------------------------------------------------------------------
def test_state_checker_round_trip():
    tok = StateChecker(state_index=2, threshold=0.5)
    restored = round_trip(tok)
    assert isinstance(restored, StateChecker)
    assert restored.state_index == 2
    assert restored.threshold == 0.5
    # state_value is transient (per-evaluation); must be None after restore
    assert restored.state_value is None
    # Set state and verify function works
    restored.set_state_value(np.array([0.3, 0.7]))
    out = restored.function(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert np.allclose(out, np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# DiscreteAction
# ---------------------------------------------------------------------------
def test_discrete_action_round_trip():
    tok = DiscreteAction(value=3)
    restored = round_trip(tok)
    assert isinstance(restored, DiscreteAction)
    assert restored.name == "a_4"
    assert int(restored.value[0]) == 3


# ---------------------------------------------------------------------------
# Pickle a list of Tokens (Library use case)
# ---------------------------------------------------------------------------
def test_mixed_token_list_round_trip():
    """A typical Library token list: input vars, ops, const placeholder."""
    tokens = [
        Token(function=None, name="x1", arity=0, complexity=1, input_var=0),
        Token(function=None, name="x2", arity=0, complexity=1, input_var=1),
        function_map["add"],
        function_map["mul"],
        function_map["log"],
        function_map["indicator"],
        PlaceholderConstant(),
        HardCodedConstant(value=2.71),
    ]
    restored = pickle.loads(pickle.dumps(tokens))
    assert len(restored) == len(tokens)
    for orig, r in zip(tokens, restored):
        assert orig.name == r.name
        assert orig.arity == r.arity
        assert orig.complexity == r.complexity
