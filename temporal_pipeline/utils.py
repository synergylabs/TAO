"""
Author: Prasoon Patidar
Created: 22nd Apr 2022

Custom utils for context training and prediction
"""

from datetime import datetime
import numpy as np
import jstyleson


def time_diff(t_start, t_end):
    """
    Get time diff in secs

    Parameters:
        t_start(datetime)               : Start time
        t_end(datetime)                 : End time

    Returns:
        t_diff(int)                     : time difference in secs
    """

    return (t_end - t_start).seconds + np.round((t_end - t_start).microseconds / 1000000, 3)


def get_config_from_json(jsonfile):
    """
    Get config from a commented json file to dictionary

    :param jsonfile: file to extract default config from
    :return: dict with config based on json file
    """
    config_dict = None
    try:
        config_dict = jstyleson.load(open(jsonfile,'r'))
    except:
        print("Unable to load default config file, exiting")
        exit(1)

    return config_dict


def activity_vec_to_activities(activity_vector,activity_labels):
    '''
    This function converts raw activity vector to activities
    '''
    try:
        assert(len(activity_labels)==len(activity_vector))
    except AssertionError:
        print(f"Wrong sized label set({activity_labels}) for activity vector, {activity_vector}")

    activity_set = np.array(activity_labels)[np.where(activity_vector)[0]]
    # for i in range(len(activity_labels)):
    #     if activity_vector[i]==1:
    #         activity_set.append(activity_labels[i])
    if len(activity_set)==0:
        return 'None'
    return ','.join(activity_set)



def get_activity_vector(activity_set,activity_labels):
    '''
    This function converts activity set to raw activity_vector
    '''
    activity_vec = np.zeros(len(activity_labels))
    if activity_set=='None' or activity_set=='':
        return activity_vec
    for activity in activity_set:
        activity_vec[activity_labels.index(activity)] = 1.
    return activity_vec


class CustomClusteringError(Exception):
    pass