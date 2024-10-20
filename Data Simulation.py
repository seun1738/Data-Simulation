"""Infectious disease dynamics simulation module.

This module simulates the infection state of each individual in the population
using the Bernoulli distribution where where all individuals are assumed to be
in either the susceptible state 0, or the infected state 1, for each week in
the study. We group each inidividual into households of sizes 3, 4 or 5
inidivuals
"""

# Standard library imports required for the algorithm
from scipy.stats import bernoulli
import math as math
import pandas as pd
import random

# The parameter alpha denotes the community acquistion infection
alpha = 0.5

# The parameter beta denotes the within-household acquisition infection
beta = 0.1

# The parameter gamma denotes the rate of recovery from the disease infection
gamma = 0.8

# The number of individuals in the entire population
number_of_individuals = 20

# The number of discrete time observations made on the infection state of each
# indvidual
number_of_simulations = 5

# Creating an empty list that will contain the infection state of each
# individual at the beginning of the observation
initial_data = []

# Randomly generating the infection state of each inidividual at the beginning
# of the observation
for individual in range(number_of_individuals):
    initial_data.append(random.randint(0, 1))

# Creating an empty list that will contain simulated infection state of each
# individual across the entire observation period
simulated_data = []

start = 0

# Grouping each inidividual into a certain household
while start < number_of_individuals:
    # Setting the number of individuals in a particular household
    size = random.randint(3, 5)
    # Assigning each along with their initial infection state into a certain
    # household
    simulated_data.append(initial_data[start:start+size])
    start += size

# If the number of individuals in the last household is smaller than 3 then
# the alogorithm needs to be rerun
if len(simulated_data[-1]) < 3:
    raise ValueError(
        'The last hosuehold shoulld not have less than 3 individuals')


# The number of households in the population
number_of_households = len(simulated_data)

# Creating an empty list that will contain the number of individuals in each
# household
household_sizes = []

# Iterating over the size of each household
for size in range(number_of_households):
    household_sizes.append(len(simulated_data[size]))

# Simulating the infectious disease dynamics in each household at all discrete
# time points
for household in range(number_of_households):
    index = 0
    # Simulating the infectious disease dynamics for each individual in the
    # household at all observation time points
    for simulations in range(number_of_simulations):

        # Creating an empty list containing the infection state data of
        # each individual at a certain time
        infection_state_at_certain_time_step = []

        # Iterating the present infection state of each individual in the
        # household
        for data in simulated_data[household][
                simulations*household_sizes[household]:]:
            # If the individual is susceptible
            if data == 0:
                infection_state_at_certain_time_step.append(
                    bernoulli.rvs(1-math.exp(-alpha-beta*[
                        simulated_data[
                            household][i] for i in range(
                                len(simulated_data[
                                    household])) if i != index][
                                        simulations*household_sizes[
                                            household]:].count(1))))
            # If the individual is infected
            else:
                infection_state_at_certain_time_step.append(
                    bernoulli.rvs(1-math.exp(-gamma)))
            index += 1
        # Appending the infection state of all inidividuals at a particular
        # time into the full population data
        for state in infection_state_at_certain_time_step:
            simulated_data[household].append(state)


# Creating column names of each hosuehold pandas dataframe
column_names = []
for columns in range(number_of_individuals):
    column_names.append('Individual ' + str(columns+1))

# Creating row names of each household pandas dataframe
row_names = []
for rows in range(number_of_simulations+1):
    row_names.append('Discrite Time, t=' + str(rows))

# Creating an empty list that will contain the dataframes of each household
# infection disease dynamics data
household_data = []

# Iterating over each household data and sorting out the infection state of
# each individual in the household
for house in range(number_of_households):
    simulated_data[house] = [simulated_data[
        house][i:i + household_sizes[house]] for i in range(
            0, len(simulated_data[house]), household_sizes[house])]

    # Creating a pandas dataframe for a particular household data
    household_data.append(pd.DataFrame(
        simulated_data[house],
        columns=[column_names[i] for i in range(household_sizes[house])]))

# Assigning row names to each household dataframe
for household in range(number_of_households):
    household_data[household].index = row_names

# Creating CSV files for each household data
for file in range(number_of_households):
    household_data[file].to_csv(f'household_data_{file+1}', index=False)
