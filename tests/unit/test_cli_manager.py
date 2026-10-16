"""Fast regression tests for pure-Python CLI argument parsing."""

import argparse

import pytest

from modules.utils.cli_manager import str2bool

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("value", ["yes", "true", "t", "y", "1", "TRUE", "Yes"])
def test_str2bool_accepts_true_values(value):
    assert str2bool(value) is True


@pytest.mark.parametrize("value", ["no", "false", "f", "n", "0", "FALSE", "No"])
def test_str2bool_accepts_false_values(value):
    assert str2bool(value) is False


@pytest.mark.parametrize("value", [True, False])
def test_str2bool_preserves_boolean_inputs(value):
    assert str2bool(value) is value


@pytest.mark.parametrize("value", ["", "maybe", "2", "none"])
def test_str2bool_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="Boolean value expected"):
        str2bool(value)
