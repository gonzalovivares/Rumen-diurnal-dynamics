import numpy as np
import pandas as pd
from numba import njit


pCO2 = 0.65         # Partial pressure of CO2 in the rumen.
CCAD = 150.0        # Concentration of ions in the rumen.
CAc  = 0.060        # Concentration of Acetate in the rumen.
CPr  = 0.025        # Concentration of Propionate in the rumen.
CBu  = 0.0125       # Concentration of Butyrate in the rumen.
CVl  = 0.0025       # Concentration of Valerate and other VFA in the rumen.
CLa  = 0.050        # Concentration of Lactic acid in the rumen.
CAm  = 0.025        # Concentration of Ammonia in the rumen.

ph_inputs_values = np.array([pCO2,          # Partial pressure of CO2 in the rumen.
                             CCAD,          # Concentration of ions in the rumen.
                             CAc,           # Concentration of Acetate in the rumen.
                             CPr,           # Concentration of Propionate in the rumen.
                             CBu,           # Concentration of Butyrate in the rumen.
                             CVl,           # Concentration of Valerate and other VFA in the rumen.
                             CLa,           # Concentration of Lactic acid in the rumen.
                             CAm])          # Concentration of Ammonia in the rumen.

@njit
def secant_method(f, x0, x1, ph_inputs, tol=0.5e-4, max_iterations=8000):
    for n in range(max_iterations):
        # Calculate function values at endpoints
        y0, y1 = f(x0, ph_inputs), f(x1, ph_inputs)

        # Calculate next root approximation
        xn = x1 - y1 * ((x1 - x0) / (y1 - y0))

        # Check tolerance condition
        if abs(y1) < tol:
            return np.array([xn, n + 1])  # n + 1 since we start counting from 0

        # Update values for the next iteration
        x0, x1 = x1, xn

    # If max_iterations is reached without meeting tolerance, return the last approximation
    return np.array([xn, max_iterations])

@njit
def run_ph(ph_inputs_values):
    def function(h_plus, ph_inputs_values):
        # Calling model inputs from array, order of array values must align with order of ph_inputs_names
        ph_inputs_names = ['pCO2', 'CCAD', 'CAc', 'CPr', 'CBu', 'CVl', 'CLa', 'CAm']

        pCO2 = ph_inputs_values[ph_inputs_names.index('pCO2')]
        CCAD = ph_inputs_values[ph_inputs_names.index('CCAD')]
        CAc = ph_inputs_values[ph_inputs_names.index('CAc')]
        CPr = ph_inputs_values[ph_inputs_names.index('CPr')]
        CBu = ph_inputs_values[ph_inputs_names.index('CBu')]
        CBc = ph_inputs_values[ph_inputs_names.index('CVl')]
        CLa = ph_inputs_values[ph_inputs_names.index('CLa')]
        CAm = ph_inputs_values[ph_inputs_names.index('CAm')]


        # Dissociation constants of pH effects.
        kac = 1.74e-5       # Dissociation constant of Acetate
        kpr = 1.35e-5       # Dissociation constant of Propionate
        kbu = 1.52e-5       # Dissociation constant of Butyrate
        kbc = 1.45e-5       # Dissociation constant of Valerate and other VFA
        kla = 1.38e-4       # Dissociation constant of Lactic acid
        kc1 = 1.82e-8       # Dissociation constant for the reaction CO2-H2CO3-
        kc2 = 2.1e-7        # Dissociation constant of HCO3--CO32-
        kam = 8.83e-10      # Dissociation constant of Ammonia

        # Alkaline effects of ions into pH
        eff_ions = CCAD + h_plus * 1000

        # Acidic effects of CO2 reactions into pH
        eff_co2_reacts = kc1 * pCO2 / h_plus * 1000 + kc1 * kc2 * pCO2 / h_plus ** 2 * 1000

        # Acidic effects of fermentation acids into pH
        eff_acids = (CAc / (h_plus / kac + 1) + CPr / (1 + h_plus / kpr) + CBu / (1 + h_plus / kbu) + CBc / (
                1 + h_plus / kbc) + CLa / (1 + h_plus / kla)) * 1000

        # Alkaline effects of ammonia into pH
        eff_am = CAm / (1 + kam / h_plus) * 1000

        return eff_ions - eff_co2_reacts - eff_acids + eff_am

    solution = secant_method(function, 10 ** (-7.5), 10 ** (-7.5) * 1.1, ph_inputs_values)
    pH = - np.log10(solution[0])
    return pH

print('Simulated rumen pH value: ', run_ph(ph_inputs_values))
