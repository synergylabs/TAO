from .directLabeler import directLabeler
from .ontolistLabeler import ontolistLabeler
from .ontologyLabeler import ontologyLabeler
from .newontolistLabeler import newontolistLabeler
from .manualLabeler import manualontolistLabeler
from .ontoconv_directLabeler import ontoconvdirectLabeler


def fetch_labeler(labeler_name, logger):
    """
    This function fetches require representation model based on model name in config
    :param labeler_name: name of RE labeler
    :param logger: logging object
    :return:
    """
    labeler = None
    if labeler_name == 'direct':
        labeler = directLabeler

    if labeler_name == 'ontolist':
        labeler = ontolistLabeler

    if labeler_name == 'newontolist':
        labeler = newontolistLabeler

    if labeler_name == 'manual':
        labeler = manualontolistLabeler

    if labeler_name == 'ontology':
        labeler = ontologyLabeler

    if labeler_name == 'onto_conv':
        labeler = ontoconvdirectLabeler

    if labeler is None:
        logger.info(f"Unable to get context representation labeler {labeler_name}. Exiting...")
        exit(1)

    return labeler
