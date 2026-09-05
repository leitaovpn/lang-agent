"""内置演示工具行为测试。"""
import pytest

from lang_agent.core.tools import calculator, string_len, string_reverse


def test_calculator_basic():
    assert calculator("(3+5)*7") == "56"


def test_calculator_float_division():
    assert calculator("7/2") == "3.5"


def test_calculator_rejects_non_math_input():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('ls')")


def test_calculator_rejects_bad_syntax():
    with pytest.raises(ValueError):
        calculator("1+")


def test_string_reverse():
    assert string_reverse("abc") == "cba"


def test_string_len():
    assert string_len("abc") == 3
