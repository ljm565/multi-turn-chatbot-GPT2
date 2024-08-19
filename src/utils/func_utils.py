from utils import LOGGER, colorstr



def print_samples(target, prediction):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('GT        : ') + target)
    LOGGER.info(colorstr('Prediction: ') + prediction)
    LOGGER.info('-'*100 + '\n')