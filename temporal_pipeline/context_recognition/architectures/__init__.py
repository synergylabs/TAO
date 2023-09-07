from .CNN_AE import CNN_AE_Network
from .FullyConnected_AE import FCN_AE_Network
from .Temporal_AE import TAE
from .LSTM_AE import LSTM_AE_Network


def fetch_re_model(model_name, logger):
    """
    This function fetches require representation model based on model name in config
    :param model_name: name of RE model
    :param logger: logging object
    :return:
    """
    model_re = None
    if model_name=='FCN':
        model_re = FCN_AE_Network
    elif model_name=='CNN':
        model_re  = CNN_AE_Network
    elif model_name=='LSTM':
        model_re = LSTM_AE_Network
    elif model_name=='TAE':
        model_re = TAE



    if model_re is None:
        logger.info(f"Unable to get representation learning model {model_name}. Exiting...")
        exit(1)

    return model_re



