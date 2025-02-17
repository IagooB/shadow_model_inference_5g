import os
import logging
from configuracion import exper_config

#### LOG ####
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(exper_config["prefijo_save"]):
    os.makedirs(exper_config["prefijo_save"])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'execution.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)