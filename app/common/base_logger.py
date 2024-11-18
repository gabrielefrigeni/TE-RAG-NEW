import logging
import sys

def get_logger(
    name: str = "__name__",
) -> logging.Logger:
    """
    Initialize and return a logger object.
    """

    logger = logging.getLogger(name)
    logging.basicConfig(
        format="[%(levelname)s] - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    return logger
