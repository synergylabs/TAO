'''
This is finite state stress pipeline, developed based on information gathered and interpreted
from wide variety of medical journal papers. It uses rule based methods to predict
stress score based on context at week level
Developer: Prasoon Patidar
Created at: 12th May 2022
'''
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance


def get_stress_score(df_wellness_input, run_config, logger):
    """
    Gets stress score based on learned contexts at weekly level
    :param df_wellness_input: timestamped context info
    :param run_config: global config
    :param logger: logging object
    :return: weekly stress score
    """

    df_stress_scores = None

    # get week ids based on timestamp data
    df_wellness_input['datetime'] = pd.to_datetime(df_wellness_input['timestamp'], unit='s')
    df_wellness_input['week'] = df_wellness_input['datetime'].apply(lambda x: x.strftime("%Y_%V"))
    df_wellness_input['day'] = df_wellness_input['datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))

    weekly_stress_scores = []
    for week in df_wellness_input['week'].unique():
        df_week_wellness_input = df_wellness_input[df_wellness_input.week == week]
        week_context_attributes = {}
        for day in df_week_wellness_input['day'].unique():
            df_day_wellness_input = df_week_wellness_input[df_week_wellness_input.day == day]
            df_day_wellness_input['context_grp'] = (
                    df_day_wellness_input['context'] != df_day_wellness_input['context'].shift(1)).cumsum()
            day_contexts = df_day_wellness_input.groupby(['context_grp', 'context'], as_index=False).agg({
                'timestamp': ['min', 'max', lambda x: x.max() - x.min()]
            })
            day_contexts.columns = ['group', 'context', 'start', 'end', 'length']
            week_context_attributes[day] = day_contexts
        week_stress_score = get_weekly_stress_score(week_context_attributes)
        weekly_stress_scores.append([week, week_stress_score])

    df_stress_scores = pd.DataFrame(weekly_stress_scores, columns=['week', 'stress_score'])

    return df_stress_scores


def get_weekly_stress_score(week_context_attributes):
    """
    Return stress score for the week
    :param week_context_attributes: How days context looks like
    :return: weekly stress score based on FSM
    """
    stress_score = 0.
    total_days_data_in_week = len(week_context_attributes.keys())

    # Condition 1: Context of exercising and housework existing in 57% of week: -1 stress score
    c1_exercise_count, c1_housework_count = 0.,0.
    for day in week_context_attributes.keys():
        if 'Exercising' in week_context_attributes[day].context.values:
            c1_exercise_count +=1
        if 'HouseWork' in week_context_attributes[day].context.values:
            c1_housework_count+=1

    if ((c1_exercise_count/total_days_data_in_week) >= 0.57) and ((c1_housework_count/total_days_data_in_week) >= 0.57):
        stress_score -= 1

    # Condition 2a: No context of exercising for more than 57% week(4 days if full week available): +0.5 stress score
    # Condition 2b: No context of housework  for more than 4 days(57% week): +0.5 stress score
    c2a_exercise_count = c1_exercise_count
    c2b_housework_count = c1_housework_count

    if (c2a_exercise_count/total_days_data_in_week) < 0.57:
        stress_score += 0.5
    if (c2b_housework_count/total_days_data_in_week) < 0.57:
        stress_score += 0.5

    # Condition 3: context of commute for more than an hour(on average) in a day everyday: +0.5 stress score
    c3_commute_hours = []
    for day in week_context_attributes:
        if 'Commuting' not in week_context_attributes[day].context.values:
            c3_commute_hours.append(0.)
        else:
            day_commute_values = week_context_attributes[day][week_context_attributes[day].context=='Commuting']
            day_commute_values['period'] = day_commute_values['end'] - day_commute_values['start']
            c3_commute_hours.append(day_commute_values['period'].sum())

    if np.mean(c3_commute_hours) > (60*60): # more than 1 hours of average commute
        stress_score +=0.5


    # condition 4a: context of sleep for less than 6 hours on more than 3 days(42% week): +1 stress score
    # condition 4b: context of sleep for less more than 8 hours on an average: -0.5 stress score
    c4_sleep_hours = []
    for day in week_context_attributes:
        if 'Sleeping' not in week_context_attributes[day].context.values:
            c4_sleep_hours.append(0.)
        else:
            day_sleep_values = week_context_attributes[day][week_context_attributes[day].context == 'Sleeping']
            day_sleep_values['period'] = day_sleep_values['end'] - day_sleep_values['start']
            c4_sleep_hours.append(day_sleep_values['period'].sum())

    if np.max(sorted(c4_sleep_hours)[:3]) < (6 * 60 * 60):  # 4a.less than 6 hours of sleep for 3 days
        stress_score += 1

    if np.mean(c4_sleep_hours) > (8 * 60 * 60):  # 4b. more than 8 hours of sleep
        stress_score -= 0.5

    # condition 5: (More than 4 days(57%) of Exercising context for more than half an hour) &
    # (Different working hours (>40% difference) across days): +0.5 stress score
    c5_exercising_count = c1_exercise_count
    c5_exercising_hours = []
    for day in week_context_attributes:
        if 'Exercising' in week_context_attributes[day].context.values:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'Exercising']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            c5_exercising_hours.append(day_exercise_values['period'].sum())
    c5_exercise_cond = False
    if ((c5_exercising_count/total_days_data_in_week) >= 0.57) and (np.min(c5_exercising_hours) > (0.5*60*60)):
        c5_exercise_cond = True

    c5_working_periods = {}
    for day in week_context_attributes:
        if 'OfficeWork' in week_context_attributes[day].context.values:
            day_work_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_work_values['start_hour'] = pd.to_datetime(day_work_values['start'],unit='s').dt.hour
            day_work_values['end_hour'] = pd.to_datetime(day_work_values['end'], unit='s').dt.hour
            c5_working_periods[day] = np.zeros(24)
            for idx, row in day_work_values.iterrows():
                c5_working_periods[day][row['start_hour']:row['end_hour']+1] = 1

    c5_hour_differences= [0.]
    for dayA in c5_working_periods.keys():
        for dayB in c5_working_periods.keys():
            if not (dayA==dayB):
                working_hoursA = c5_working_periods[dayA]
                working_hoursB = c5_working_periods[dayB]
                diff_hours = np.sum(working_hoursA!=working_hoursB)
                same_hours = np.sum(np.logical_and(working_hoursA==working_hoursB,working_hoursA))
                if same_hours > 0.:
                    c5_hour_differences.append(diff_hours / same_hours)
                else:
                    c5_hour_differences.append(0.)

    c5_working_cond = False
    if np.max(c5_hour_differences) > 0.4:
        c5_working_cond = True

    if (c5_working_cond) & (c5_exercise_cond):
        stress_score +=0.5

    # condition 6: (No context of exercising) & (Different working hours across days): +1 stress score
    c6_exercise_count=  c1_exercise_count
    c6_working_hour_diff = c5_hour_differences
    if (c6_exercise_count==0.) & (np.mean(c5_hour_differences) > 0.4):
        stress_score += 1

    # condition 7: (More than 4 days of Exercising context for more than half an hour) &
    # (No Different working hours (>40% difference)  across days): -1 stress score
    c7_exercise_cond = c5_exercise_cond
    c7_hour_differences = c5_hour_differences

    if (c7_exercise_cond) & (np.max(c7_hour_differences) < 0.4):
        stress_score -= 1.

    # condition 8: (More than 10 hrs of works in any day) OR (50+ hours per week): +1 stress score
    c8_work_hours = []
    for day in week_context_attributes:
        if 'OfficeWork' in week_context_attributes[day].context.values:
            day_work_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_work_values['period'] = day_work_values['end'] - day_work_values['start']
            c8_work_hours.append(day_work_values['period'].sum())
        else:
            c8_work_hours.append(0.)

    if (np.max(c8_work_hours) > (10 * 60 * 60)) or (np.sum(c8_work_hours) > (50 * 60 * 60)):
        stress_score += 1

    # condition 9a: (More than 40% shift in working hours in max gap): +1 stress score
    # condition 9b: (More than 40% shift in working hours in 2 or more consecutive days): +1 stress score
    # Not modeling due to over complexity

    # condition 10a: More than 15 mins exercising every day: -0.5 stress score
    # condition 10b: More than 1 hour housework every day: -0.5 stress score
    c10a_exercising_hours = []
    for day in week_context_attributes:
        if 'Exercising' in week_context_attributes[day].context.values:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'Exercising']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            c10a_exercising_hours.append(day_exercise_values['period'].sum())
        else:
            c10a_exercising_hours.append(0.)

    c10b_housework_hours = []
    for day in week_context_attributes:
        if 'HouseWork' not in week_context_attributes[day].context.values:
            c10b_housework_hours.append(0.)
        else:
            day_housework = week_context_attributes[day][week_context_attributes[day].context == 'HouseWork']
            day_housework['period'] = day_housework['end'] - day_housework['start']
            c10b_housework_hours.append(day_housework['period'].sum())

    if np.min(c10a_exercising_hours) > 15 * 60:
        stress_score -=0.5

    if np.min(c10b_housework_hours) > 60 * 60:
        stress_score -= 0.5



    # condition 11a: Continuous 30 mins exercise, five times a week: -1 stress score
    # condition 11b: 15 mins exercise twice, five times a week: -1 stress score
    # condition 11c: 10 mins exercise thrice, five times a week: -1 stress score

    c11_day_cond = []
    for day in week_context_attributes.keys():
        if 'Exercising' not in week_context_attributes[day].context.values:
            c11_day_cond.append(False)
        else:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            day_num_exercises = day_exercise_values.shape[0]
            day_min_continuous_exercise = day_exercise_values['period'].min()
            if (day_num_exercises >=1) and (day_min_continuous_exercise > 30*60):
                c11_day_cond.append(True)
            elif (day_num_exercises >=2) and (day_min_continuous_exercise > 15*60):
                c11_day_cond.append(True)
            elif (day_num_exercises >=3) and (day_min_continuous_exercise > 10*60):
                c11_day_cond.append(True)
            else:
                c11_day_cond.append(False)

    if (np.sum(c11_day_cond)/len(c11_day_cond)) > 0.7: # 5+ days in a week
        stress_score -= 1

    # condition 12: Working hours more than 8 for more than 5 days: +1 stress score
    c12_work_hours = c8_work_hours

    if np.percentile(c12_work_hours,0.7) > (8 * 60 * 60):
        stress_score += 1

    # condition 13a: Inactivity/Amusement for more than 2 hours everyday: -0.5 stress score
    # condition 13b: Inactivity/Amusement/Housework for more than 5 hours atleast two days: -0.5 stress score
    c13a_inactivity_amusement_hours = []
    for day in week_context_attributes:
        day_13a_hours = week_context_attributes[day][week_context_attributes[day].context.isin(['Inactivity','Amusement'])]
        if day_13a_hours.shape[0] > 0.:
            c13a_inactivity_amusement_hours.append((day_13a_hours['end'] - day_13a_hours['start']).sum())
        else:
            c13a_inactivity_amusement_hours.append(0.)

    c13b_inactivity_amusement_housework_hours = []
    for day in week_context_attributes:
        day_13b_hours = week_context_attributes[day][week_context_attributes[day].context.isin(['Inactivity','Amusement','HouseWork'])]
        if day_13b_hours.shape[0] > 0.:
            c13b_inactivity_amusement_housework_hours.append((day_13b_hours['end'] - day_13b_hours['start']).sum())
        else:
            c13b_inactivity_amusement_housework_hours.append(0.)

    if np.min(c13a_inactivity_amusement_hours) > (2 * 60 * 60):
        stress_score -= 0.5
    if np.percentile(c13b_inactivity_amusement_housework_hours, 30) > (5 * 60 * 60):
        stress_score -= 0.5

    # condition 14a: Working hours between 12am and 8am for any day: +1 stress score
    # condition 14b: Working hours between 12am and 8am for more than 4 days: +1 stress score

    c14_working_periods = c5_working_periods
    c14_night_work = []
    for day in c14_working_periods.keys():
        if np.sum(c14_working_periods[day][:8]) > 1:
            c14_night_work.append(True)
        else:
            c14_night_work.append(False)

    if np.sum(c14_night_work) >=1:
        stress_score +=1
    if np.sum(c14_night_work) >=4:
        stress_score += 1

    return stress_score

def get_daily_stress_score(week_context_attributes):
    """
    Return stress score for the week, and bifurcation across days
    :param week_context_attributes: How days context looks like
    :return: weekly stress score based on FSM
    """
    stress_score = 0.
    stress_score_days = {day:0. for day in week_context_attributes}
    total_days_data_in_week = len(week_context_attributes.keys())
    # Condition 1: Context of exercising and housework existing in 57% of week: -1 stress score
    c1_exercise_count, c1_housework_count = 0.,0.
    c1_eday, c1_hday = [],[]
    for day in week_context_attributes.keys():
        if 'Exercising' in week_context_attributes[day].context.values:
            c1_exercise_count +=1
            c1_eday.append(day)
        if 'HouseWork' in week_context_attributes[day].context.values:
            c1_housework_count+=1
            c1_hday.append(day)

    if ((c1_exercise_count/total_days_data_in_week) >= 0.57) and ((c1_housework_count/total_days_data_in_week) >= 0.57):
        stress_score -= 1
        for day in c1_eday:
            stress_score_days[day]-=(0.5)/total_days_data_in_week
        for day in c1_hday:
            stress_score_days[day]-=(0.5)/total_days_data_in_week
    stress_score_days

    c2a_exercise_count = c1_exercise_count
    c2b_housework_count = c1_housework_count

    if (c2a_exercise_count/total_days_data_in_week) < 0.57:
        for day in stress_score_days:
            if 'Exercising' not in week_context_attributes[day].context.values:
                stress_score_days[day] += (0.5/total_days_data_in_week)
        stress_score += 0.5
    if (c2b_housework_count/total_days_data_in_week) < 0.57:
        for day in stress_score_days:
            if 'HouseWork' not in week_context_attributes[day].context.values:
                stress_score_days[day] += (0.5/total_days_data_in_week)
        stress_score += 0.5

    stress_score_days

    # Condition 3: context of commute for more than an hour(on average) in a day everyday: +0.5 stress score
    c3_commute_hours = []
    c3_commute_hours_days = {day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'Commuting' not in week_context_attributes[day].context.values:
            c3_commute_hours.append(0.)
        else:
            day_commute_values = week_context_attributes[day][week_context_attributes[day].context=='Commuting']
            day_commute_values['period'] = day_commute_values['end'] - day_commute_values['start']
            c3_commute_hours_days[day]=day_commute_values['period'].sum()
            c3_commute_hours.append(day_commute_values['period'].sum())

    if np.mean(c3_commute_hours) > (60*60): # more than 1 hours of average commute
        stress_score +=0.5
        total_commute_hours = np.sum(c3_commute_hours)
        for day in stress_score_days:
            stress_score_days[day] +=  (0.5/total_days_data_in_week)*(c3_commute_hours_days[day]/total_commute_hours)

    stress_score_days

    # condition 4a: context of sleep for less than 6 hours on more than 3 days(42% week): +1 stress score
    # condition 4b: context of sleep for less more than 8 hours on an average: -0.5 stress score
    c4_sleep_hours = []
    c4_sleep_hours_days = {day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'Sleeping' not in week_context_attributes[day].context.values:
            c4_sleep_hours.append(0.)
        else:
            day_sleep_values = week_context_attributes[day][week_context_attributes[day].context == 'Sleeping']
            day_sleep_values['period'] = day_sleep_values['end'] - day_sleep_values['start']
            c4_sleep_hours_days[day] = day_sleep_values['period'].sum()
            c4_sleep_hours.append(day_sleep_values['period'].sum())

    if np.max(sorted(c4_sleep_hours)[:3]) < (6 * 60 * 60):  # 4a.less than 6 hours of sleep for 3 days
        stress_score += 1
        if c4_sleep_hours_days[day]<(6 * 60 * 60):
            stress_score_days[day]+=(1/total_days_data_in_week)

    if np.mean(c4_sleep_hours) > (8 * 60 * 60):  # 4b. more than 8 hours of sleep
        stress_score -= 0.5
        if c4_sleep_hours_days[day]>(6 * 60 * 60):
            stress_score_days[day]-=(0.5/total_days_data_in_week)
    stress_score_days




    # condition 5: (More than 4 days(57%) of Exercising context for more than half an hour) &
    # (Different working hours (>40% difference) across days): +0.5 stress score
    c5_exercising_count = c1_exercise_count
    c5_exercising_hours = []
    c5_exercising_day_hours = {day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'Exercising' in week_context_attributes[day].context.values:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'Exercising']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            c5_exercising_hours.append(day_exercise_values['period'].sum())
            c5_exercising_day_hours[day]=day_exercise_values['period'].sum()

    c5_exercise_cond = False
    if ((c5_exercising_count/total_days_data_in_week) >= 0.57) and (np.min(c5_exercising_hours) > (0.5*60*60)):
        c5_exercise_cond = True


    c5_working_periods = {}
    for day in week_context_attributes:
        if 'OfficeWork' in week_context_attributes[day].context.values:
            day_work_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_work_values['start_hour'] = pd.to_datetime(day_work_values['start'],unit='s').dt.hour
            day_work_values['end_hour'] = pd.to_datetime(day_work_values['end'], unit='s').dt.hour
            c5_working_periods[day] = np.zeros(24)
            for idx, row in day_work_values.iterrows():
                c5_working_periods[day][row['start_hour']:row['end_hour']+1] = 1


    c5_hour_differences= [0.]
    c5_hour_differences_day = {day:[0.] for day in stress_score_days}
    for dayA in c5_working_periods.keys():
        dayA_differences=[]
        for dayB in c5_working_periods.keys():
            if not (dayA==dayB):
                working_hoursA = c5_working_periods[dayA]
                working_hoursB = c5_working_periods[dayB]
                diff_hours = np.sum(working_hoursA!=working_hoursB)
                same_hours = np.sum(np.logical_and(working_hoursA==working_hoursB,working_hoursA))
                if same_hours > 0.:
                    c5_hour_differences.append(diff_hours / same_hours)
                    dayA_differences.append(diff_hours / same_hours)
                else:
                    c5_hour_differences.append(0.)
        c5_hour_differences_day[dayA]+=dayA_differences

    c5_working_cond = False
    if np.max(c5_hour_differences) > 0.4:
        c5_working_cond = True
        min_hour_start = min([c5_working_periods[day_x].argmin() for day_x in c5_working_periods])
        min_hour_end = min([c5_working_periods[day_x].argmin() for day_x in c5_working_periods])

    if (c5_working_cond) & (c5_exercise_cond):
        stress_score +=0.5
        for day in stress_score_days:
            if c5_exercising_day_hours[day]<np.max(c5_exercising_hours):
                    stress_score_days[day]+=(0.5/total_days_data_in_week)

        for day in stress_score_days:
            if c5_exercising_day_hours[day]<np.max(c5_exercising_hours):
                    stress_score_days[day]+=(0.25/total_days_data_in_week)
            stress_score_days[day]+=(0.25/total_days_data_in_week)*(np.max(c5_hour_differences_day[day])/np.max(c5_hour_differences))

    stress_score_days

    # condition 6: (No context of exercising) & (Different working hours across days): +1 stress score
    c6_exercise_count=  c1_exercise_count
    c6_working_hour_diff = c5_hour_differences
    if (c6_exercise_count==0.) & (np.mean(c5_hour_differences) > 0.4):
        stress_score += 1
        for day in stress_score_days:
            stress_score_days[day]+=1/total_days_data_in_week
    stress_score_days

    # condition 7: (More than 4 days of Exercising context for more than half an hour) &
    # (No Different working hours (>40% difference)  across days): -1 stress score
    c7_exercise_cond = c5_exercise_cond
    c7_hour_differences = c5_hour_differences

    if (c7_exercise_cond) & (np.max(c7_hour_differences) < 0.4):
        stress_score -= 1.
        for day in stress_score_days:
            stress_score_days[day]-=1/total_days_data_in_week

    # condition 8: (More than 10 hrs of works in any day) OR (50+ hours per week): +1 stress score
    c8_work_hours = []
    c8_work_hours_day= {day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'OfficeWork' in week_context_attributes[day].context.values:
            day_work_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_work_values['period'] = day_work_values['end'] - day_work_values['start']
            c8_work_hours.append(day_work_values['period'].sum())
            c8_work_hours_day[day]=day_work_values['period'].sum()
        else:
            c8_work_hours.append(0.)

    if (np.max(c8_work_hours) > (10 * 60 * 60)) or (np.sum(c8_work_hours) > (50 * 60 * 60)):
        stress_score += 1
        for day in stress_score_days:
            if c8_work_hours_day[day]>10*60*60:
                stress_score_days[day]+=1/total_days_data_in_week

    stress_score_days

    # condition 10a: More than 15 mins exercising every day: -0.5 stress score
    # condition 10b: More than 1 hour housework every day: -0.5 stress score
    c10a_exercising_hours = []
    c10a_exercising_hours_day={day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'Exercising' in week_context_attributes[day].context.values:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'Exercising']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            c10a_exercising_hours.append(day_exercise_values['period'].sum())
            c10a_exercising_hours_day[day]=day_exercise_values['period'].sum()
        else:
            c10a_exercising_hours.append(0.)

    c10b_housework_hours = []
    c10b_housework_hours_day  ={day:0. for day in stress_score_days}
    for day in week_context_attributes:
        if 'HouseWork' not in week_context_attributes[day].context.values:
            c10b_housework_hours.append(0.)
        else:
            day_housework = week_context_attributes[day][week_context_attributes[day].context == 'HouseWork']
            day_housework['period'] = day_housework['end'] - day_housework['start']
            c10b_housework_hours.append(day_housework['period'].sum())
            c10b_housework_hours_day[day]=day_housework['period'].sum()

    if np.min(c10a_exercising_hours) > 15 * 60:
        stress_score -=0.5
        for day in stress_score_days:
            stress_score_days[day]-=(0.5/total_days_data_in_week)


    if np.min(c10b_housework_hours) > 60 * 60:
        stress_score -= 0.5
        for day in stress_score_days:
            stress_score_days[day]-=(0.5/total_days_data_in_week)
    stress_score_days


    # condition 11a: Continuous 30 mins exercise, five times a week: -1 stress score
    # condition 11b: 15 mins exercise twice, five times a week: -1 stress score
    # condition 11c: 10 mins exercise thrice, five times a week: -1 stress score

    c11_day_cond = {day:False for day in stress_score_days}
    for day in week_context_attributes.keys():
        if 'Exercising' not in week_context_attributes[day].context.values:
            continue
        else:
            day_exercise_values = week_context_attributes[day][week_context_attributes[day].context == 'OfficeWork']
            day_exercise_values['period'] = day_exercise_values['end'] - day_exercise_values['start']
            day_num_exercises = day_exercise_values.shape[0]
            day_min_continuous_exercise = day_exercise_values['period'].min()
            if (day_num_exercises >=1) and (day_min_continuous_exercise > 30*60):
                c11_day_cond[day]=True
            elif (day_num_exercises >=2) and (day_min_continuous_exercise > 15*60):
                c11_day_cond[day]=True
            elif (day_num_exercises >=3) and (day_min_continuous_exercise > 10*60):
                c11_day_cond[day]=True
            else:
                c11_day_cond[day]=False

    if (sum(c11_day_cond.values())/len(c11_day_cond)) > 0.7: # 5+ days in a week
        stress_score -= 1
        for day in stress_score_days:
            if c11_day_cond[day]:
                stress_score_days[day]-=1/total_days_data_in_week

    # condition 12: Working hours more than 8 for more than 5 days: +1 stress score
    c12_work_hours = c8_work_hours

    if np.percentile(c12_work_hours,0.7) > (8 * 60 * 60):
        stress_score += 1
        for day in stress_score_days:
            if c8_work_hours_day[day]>8*60*60:
                stress_score_days[day]+=1/total_days_data_in_week
    stress_score_days

    # condition 13a: Inactivity/Amusement for more than 2 hours everyday: -0.5 stress score
    # condition 13b: Inactivity/Amusement/Housework for more than 5 hours atleast two days: -0.5 stress score
    c13a_inactivity_amusement_hours = []
    c13a_inactivity_amusement_hours_day={day:0. for day in stress_score_days}
    for day in week_context_attributes:
        day_13a_hours = week_context_attributes[day][week_context_attributes[day].context.isin(['Inactivity','Amusement'])]
        if day_13a_hours.shape[0] > 0.:
            c13a_inactivity_amusement_hours.append((day_13a_hours['end'] - day_13a_hours['start']).sum())
            c13a_inactivity_amusement_hours_day[day]=(day_13a_hours['end'] - day_13a_hours['start']).sum()
        else:
            c13a_inactivity_amusement_hours.append(0.)

    c13b_inactivity_amusement_housework_hours = []
    c13b_inactivity_amusement_housework_hours_day = {day:0. for day in stress_score_days}
    for day in week_context_attributes:
        day_13b_hours = week_context_attributes[day][week_context_attributes[day].context.isin(['Inactivity','Amusement','HouseWork'])]
        if day_13b_hours.shape[0] > 0.:
            c13b_inactivity_amusement_housework_hours.append((day_13b_hours['end'] - day_13b_hours['start']).sum())
            c13b_inactivity_amusement_housework_hours_day[day] = (day_13b_hours['end'] - day_13b_hours['start']).sum()
        else:
            c13b_inactivity_amusement_housework_hours.append(0.)

    if np.min(c13a_inactivity_amusement_hours) > (2 * 60 * 60):
        stress_score -= 0.5
        for day in stress_score_days:
            stress_score_days[day]-=0.5/total_days_data_in_week
    if np.percentile(c13b_inactivity_amusement_housework_hours, 30) > (5 * 60 * 60):
        stress_score -= 0.5
        for day in stress_score_days:
            if c13b_inactivity_amusement_housework_hours_day[day]> (5 * 60 * 60):
                stress_score_days[day]-=0.5/total_days_data_in_week
    stress_score_days


    c14_working_periods = c5_working_periods
    c14_night_work = []
    c14_night_work_days = {day:False for day in stress_score_days}
    for day in c14_working_periods.keys():
        if np.sum(c14_working_periods[day][:8]) > 1:
            c14_night_work.append(True)
            c14_night_work_days[day]=True
        else:
            c14_night_work.append(False)

    if np.sum(c14_night_work) >=1:
        stress_score +=1
        for day in stress_score_days:
            if c14_night_work_days[day]:
                stress_score_days[day]+=1./total_days_data_in_week
    if np.sum(c14_night_work) >=4:
        stress_score += 1
        for day in stress_score_days:
            if c14_night_work_days[day]:
                stress_score_days[day]+=1./total_days_data_in_week
    stress_score_days
    return stress_score, stress_score_days