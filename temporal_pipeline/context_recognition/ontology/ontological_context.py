'''
Wrapper file to read ontologies and get instantaneous contexts
'''

import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import glob
import os
from packaging import version
from owlready2 import get_ontology,sync_reasoner
import rdflib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters


def load_rdf3_ontology(ontology_file):
    '''

    Read ontologies from single source in owl format

    :param ontology_file: source owl file for ontology
    :return: ontology graph
    '''
    ontology_owl = get_ontology(ontology_file).load()
    with ontology_owl:
        sync_reasoner()

    return ontology_owl


def load_rdf3_ontology(ontology_n3_file, ontology_ttl_file):
    '''

    Read ontologies from multiple sources in rdf format

    :param ontology_n3_file: source n3 file for ontology
    :param ontology_ttl_file: source ttl file for ontology
    :return: ontology graph from both files
    '''
    ontology_graph = rdflib.Graph()
    ontology_graph.parse(ontology_n3_file)
    ontology_graph.serialize(ontology_ttl_file)

    return ontology_graph


def search_context_for_Activity_onto_sparql(activity, ontology_graph, step=2):
    '''
    Get context based on activity list from rdf ontology graph
    :param activity: list of comma seperated activities for context retrieval (i.e ="ns1:Sitting")
    :param ontology_graph: rdf graph object to query
    :param step: no. of parallel/sequential activities to search across
    :return: context name
    '''
    results = ontology_graph.query(
        f"""
    Select ?Context where {{ 
        ?Context owl:equivalentClass ?x . 
        ?x ?r ?y . 
        ?y ?a ?b .
        ?b rdf:first/rdf:rest* ?d . 
        ?d ?e {activity} .
    }}
    """
    )
    # print(results, activity)
    # for i in results:
    #     print(i)
    return results




