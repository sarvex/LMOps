import os
import logging

from transformers.trainer_callback import TrainerCallback


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{data_dir}/log.txt')
    file_handler.setFormatter(log_format)

    logger.handlers = [console_handler, file_handler]

    return logger


logger = _setup_logger()


class LoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_world_process_zero:
            logger.info(logs)
