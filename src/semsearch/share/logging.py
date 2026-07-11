import logging.config

from semsearch.share.config import LogLevel


def configure_logging(level: LogLevel) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "semsearch": {
                    "handlers": ["stderr"],
                    "level": level,
                    "propagate": False,
                }
            },
        }
    )
