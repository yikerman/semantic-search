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
                "semsearch": {"level": level},
                # uvicorn installs its own handlers with a different format;
                # strip them so its records render through the root handler
                # like everything else.
                "uvicorn": {"level": "INFO", "propagate": True},
                "uvicorn.access": {"level": "INFO", "propagate": True},
            },
            "root": {"handlers": ["stderr"], "level": "WARNING"},
        }
    )
