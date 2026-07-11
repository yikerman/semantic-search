import logging
import re

from semsearch.share.logging import configure_logging


def test_configure_logging_writes_one_formatted_record_to_stderr(capsys):
    app_logger = logging.getLogger("semsearch")
    previous_handlers = app_logger.handlers.copy()
    previous_level = app_logger.level
    previous_propagate = app_logger.propagate

    try:
        configure_logging("INFO")
        configure_logging("INFO")

        logging.getLogger("semsearch.test").info("configured once")

        stderr = capsys.readouterr().err
        assert stderr.count("configured once") == 1
        assert re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4} "
            r"INFO semsearch\.test: configured once",
            stderr,
        )
    finally:
        for handler in app_logger.handlers:
            if handler not in previous_handlers:
                handler.close()
        app_logger.handlers = previous_handlers
        app_logger.setLevel(previous_level)
        app_logger.propagate = previous_propagate


def test_configure_logging_applies_selected_threshold(capsys):
    app_logger = logging.getLogger("semsearch")
    previous_handlers = app_logger.handlers.copy()
    previous_level = app_logger.level
    previous_propagate = app_logger.propagate

    try:
        configure_logging("WARNING")

        logger = logging.getLogger("semsearch.test")
        logger.info("hidden")
        logger.warning("visible")

        stderr = capsys.readouterr().err
        assert "hidden" not in stderr
        assert "visible" in stderr
    finally:
        for handler in app_logger.handlers:
            if handler not in previous_handlers:
                handler.close()
        app_logger.handlers = previous_handlers
        app_logger.setLevel(previous_level)
        app_logger.propagate = previous_propagate
