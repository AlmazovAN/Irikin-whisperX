"""Test the lightweight logging helper without loading the AI stack."""

import logging

import pytest

from modules.utils.logger import get_logger

pytestmark = pytest.mark.fast


def test_get_logger_reuses_handler():
    name = "whisperx-fast-ci-logger-test"
    logger = logging.getLogger(name)
    original_handlers = list(logger.handlers)
    try:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        first = get_logger(name)
        second = get_logger(name)

        assert first is second
        assert len(first.handlers) == 1
        assert first.level == logging.INFO
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in original_handlers:
            logger.addHandler(handler)


def test_get_logger_uses_default_name():
    assert get_logger().name == "Whisper-WebUI"
