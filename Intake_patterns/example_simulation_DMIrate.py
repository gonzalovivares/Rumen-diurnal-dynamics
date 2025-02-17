import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import numba
from numba import jit, njit, types
from numba.typed import List

from diffeqpy import de
import time
from datetime import timedelta

@njit
def dmi_rate_daily_scheme(state_vars, inputs, t):
    def extract_integer_decimal(x):
        integer_part = math.floor(x)
        decimal_part = x - integer_part
        return np.array([integer_part, decimal_part])
    feed = state_vars[0]
    #  print(state_vars, feed)

    input_vals = inputs
    input_names = ['drop', 'pulse_dm', 'dm_availability', 'n_meals', 'meal_len', 'n_feedings' ]

    n_meals = input_vals[input_names.index('n_meals')]
    meal_len = input_vals[input_names.index('meal_len')]
    pulse_i = input_vals[input_names.index('pulse_dm')]
    k_drop = input_vals[input_names.index('drop')]
    regime = input_vals[input_names.index('dm_availability')]
    n_feedings = input_vals[input_names.index('n_feedings')]

    # n_day, decimal_day = extract_integer_decimal(t)

    def meal_scheme(t, n_meals, meal_len, pulse_i, drop, regime, n_feedings):
        #  pulse_i = inputs['pulse_dm']
        n_day, decimal_day = extract_integer_decimal(t)
        k_drop = drop
        meal_len_extra = meal_len
        start_time_meal = meal_len * 0.4
        end_time_meal = meal_len * 0.8
        big_meal_extension = 3.05 - 0.1375 * n_meals
        n_feedings = n_feedings
        if regime == 1:
            if n_feedings == 1:
                long_meal_1 = 1
                long_meal_2 = -50
                long_meal_3 = -50
                long_meal_4 = -50
            elif n_feedings == 2:
                long_meal_1 = 1
                long_meal_2 = np.ceil(n_meals / n_feedings) + 1
                long_meal_3 = -50
                long_meal_4 = -50
            elif n_feedings == 3:
                long_meal_1 = 1
                long_meal_2 = np.ceil(n_meals / n_feedings) + 1
                long_meal_3 = np.ceil(n_meals * (2 / 3)) + 1
                long_meal_4 = -50
            elif n_feedings == 4:
                long_meal_1 = 1
                long_meal_2 = np.ceil(n_meals / n_feedings) + 1
                long_meal_3 = np.ceil(n_meals / (n_feedings / 2)) + 1
                long_meal_4 = np.ceil(n_meals / (n_feedings / 3)) + 1
            else:
                long_meal_1 = -50
                long_meal_2 = -50
                long_meal_3 = -50
                long_meal_4 = -50
        elif regime == 0:
            if n_feedings == 1:
                long_meal_1 = 1
                long_meal_2 = -50
                long_meal_3 = -50
                long_meal_4 = -50
            else:
                long_meal_1 = -50
                long_meal_2 = -50
                long_meal_3 = -50
                long_meal_4 = -50
        else:
            long_meal_1 = 1
            long_meal_2 = -50
            long_meal_3 = -50
            long_meal_4 = -50


        def meal_definition_restricted(decimal_day, n_meals, meal_len):
            if (decimal_day < 1 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (1 == long_meal_1) | (1 == long_meal_2) | (1 == long_meal_3) | (1 == long_meal_4)):
                meal_i = 1
                meal_len_extra = meal_len * big_meal_extension
            elif decimal_day < 1 / (n_meals + 1) + (meal_len / 60) / 24:
                meal_i = 1
                meal_len_extra = meal_len

            elif (decimal_day >= 1 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 2 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (2 == long_meal_1) | (2 == long_meal_2) | (2 == long_meal_3) | (2 == long_meal_4)):
                meal_i = 2
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 1 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 2 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 2
                meal_len_extra = meal_len


            elif (decimal_day >= 2 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 3 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (3 == long_meal_1) | (3 == long_meal_2) | (3 == long_meal_3) | (3 == long_meal_4)):
                meal_i = 3
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 2 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 3 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 3
                meal_len_extra = meal_len


            elif (decimal_day >= 3 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 4 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (4 == long_meal_1) | (4 == long_meal_2) | (4 == long_meal_3) | (4 == long_meal_4)):
                meal_i = 4
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 3 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 4 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 4
                meal_len_extra = meal_len


            elif (decimal_day >= 4 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 5 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (5 == long_meal_1) | (5 == long_meal_2) | (5 == long_meal_3) | (5 == long_meal_4)):
                meal_i = 5
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 4 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 5 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 5
                meal_len_extra = meal_len


            elif (decimal_day >= 5 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 6 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (6 == long_meal_1) | (6 == long_meal_2) | (6 == long_meal_3) | (6 == long_meal_4)):
                meal_i = 6
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 5 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 6 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 6
                meal_len_extra = meal_len


            elif (decimal_day >= 6 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 7 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (7 == long_meal_1) | (7 == long_meal_2) | (7 == long_meal_3) | (7 == long_meal_4)):
                meal_i = 7
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 6 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 7 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 7
                meal_len_extra = meal_len


            elif (decimal_day >= 7 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 8 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (8 == long_meal_1) | (8 == long_meal_2) | (8 == long_meal_3) | (8 == long_meal_4)):
                meal_i = 8
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 7 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 8 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 8
                meal_len_extra = meal_len


            elif (decimal_day >= 8 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 9 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (9 == long_meal_1) | (9 == long_meal_2) | (9 == long_meal_3) | (9 == long_meal_4)):
                meal_i = 9
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 8 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 9 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 9
                meal_len_extra = meal_len


            elif (decimal_day >= 9 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 10 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (10 == long_meal_1) | (10 == long_meal_2) | (10 == long_meal_3) | (10 == long_meal_4)):
                meal_i = 10
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 9 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 10 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 10
                meal_len_extra = meal_len


            elif (decimal_day >= 10 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 11 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (11 == long_meal_1) | (11 == long_meal_2) | (11 == long_meal_3) | (11 == long_meal_4)):
                meal_i = 11
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 10 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 11 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 11
                meal_len_extra = meal_len


            elif (decimal_day >= 11 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 12 / (n_meals + 1) + (meal_len * big_meal_extension / 60) / 24) & (
                    (12 == long_meal_1) | (12 == long_meal_2) | (12 == long_meal_3) | (12 == long_meal_4)):
                meal_i = 12
                meal_len_extra = meal_len * big_meal_extension
            elif (decimal_day >= 11 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 12 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 12
                meal_len_extra = meal_len



            elif (decimal_day >= 12 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 13 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 13
                meal_len_extra = meal_len


            elif (decimal_day >= 13 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 14 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 14
                meal_len_extra = meal_len


            elif (decimal_day >= 14 / (n_meals + 1) + (meal_len / 60) / 24) & (
                    decimal_day < 15 / (n_meals + 1) + (meal_len / 60) / 24):
                meal_i = 15
                meal_len_extra = meal_len

            else:
                meal_i = 1
                meal_len_extra = meal_len
            return meal_i, meal_len_extra

        # print(t, meal_i)
        meal_i, meal_len_extra = meal_definition_restricted(decimal_day, n_meals, meal_len)

        if (t >= n_day + meal_i / (n_meals + 1)) & (
                t <= (n_day + meal_i / (n_meals + 1) + (start_time_meal / 60) / 24)):
            pulse = pulse_i
            k_drop = 0.0
        elif (t > (n_day + meal_i / (n_meals + 1) + start_time_meal / 60 / 24)) & (
                t <= (n_day + meal_i / (n_meals + 1) + (meal_len_extra - meal_len * 0.2) / 60 / 24)):
            pulse = 0.0
            k_drop = 0.0
        else:
            pulse = 0.0
            k_drop = drop

        if (regime == 1.0) | ((regime == 0.0) & (n_feedings == 1)):
            if (meal_i != 1) & (meal_i != long_meal_2) & (meal_i != long_meal_3) & (meal_i != long_meal_4):
                pulse = pulse * 0.7
            else:
                pulse = pulse
        else:
            pulse = pulse

        # pulse = float(pulse)
        # k_drop = float(drop)
        return pulse, k_drop



    pulse, k_drop = meal_scheme(t, n_meals, meal_len, pulse_i, k_drop, regime, n_feedings)

    if feed <= 5.:
        k_drop = k_drop * 0.001
        feed_dt = pulse - feed * k_drop
    else:
        feed_dt = pulse - feed * k_drop

    return feed_dt


def multiple_simulation_timestep_per_row(model_inputs, model_def, initial_states, t_span):
    results = pd.DataFrame([])
    results['author_code'] = []
    results['trt_code'] = []
    results['model_code'] = []

    results['feed'] = []
    results['time'] = []

    for i in range(model_inputs.shape[0]):
        # Change serie to numpy arrays
        input_values = np.array(model_inputs.iloc[i][2:].values)
        input_names = model_inputs.iloc[i][2:].index.to_numpy()

        inputs = input_values
        typed_inputs = List()
        [typed_inputs.append(x) for x in inputs]

        typed_initial = List()
        [typed_initial.append(x) for x in initial_states]

        row_df = pd.DataFrame([])

        print('Running ', model_inputs['author_code'].iloc[i], model_inputs['trt_code'].iloc[i], len(input_names))
        start_time = time.time()


        typed_inputs = np.array(typed_inputs)
        typed_inputs = np.ravel(typed_inputs, order='K')


        prob = de.ODEProblem(model_def, typed_initial, t_span, typed_inputs)
        sol = de.solve(prob, de.CVODE_BDF(), dtmax=0.000125, saveat=0.0015)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = str(timedelta(seconds=elapsed_time))
        simulation_i = de.stack(sol.u)
        simulation_i = np.array(simulation_i)
        simulation_i = np.vstack([simulation_i, sol.t])
        simulation_i = simulation_i[:, simulation_i[-1] > 9.0]

        print('Simulation done, time spent: ', formatted_time)

        row_df['time'] = simulation_i[-1]
        row_df['feed'] = simulation_i[0]
        row_df['author_code'] = model_inputs['author_code'].iloc[i]
        row_df['trt_code'] = model_inputs['trt_code'].iloc[i]

        results = pd.concat([results, row_df])



    return results


initial_states = [5.0]
t_span = [0.,10.]

model_inputs = pd.DataFrame([], columns=['author_code', 'trt_code', 'drop', 'pulse_dm', 'dm_availability', 'n_meals', 'meal_len', 'n_feedings' ])

model_inputs.loc[0, 'author_code'] = 'example_1'
model_inputs.loc[0, 'trt_code'] = 'adlib_8meals'
model_inputs.loc[0, 'drop'] = 250.0
model_inputs.loc[0, 'pulse_dm'] = 1700.0
model_inputs.loc[0, 'dm_availability'] = 0.0
model_inputs.loc[0, 'n_meals'] = 8.0
model_inputs.loc[0, 'meal_len'] = 35.0
model_inputs.loc[0, 'n_feedings'] = 2.0

print(model_inputs.to_string())
results = multiple_simulation_timestep_per_row(model_inputs , dmi_rate_daily_scheme, initial_states, t_span)

plt.plot((results.time - 9.0) * 24.0, results.feed, color = 'black')
plt.xlabel('Time, (hour)')
plt.ylabel('DMI rate, (kg/d)')

plt.show()
