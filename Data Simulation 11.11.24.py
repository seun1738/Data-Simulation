"""Infectious disease dynamics simulation module.

This module simulates the infection state of each individual in the population
using the Bernoulli distribution where where all individuals are assumed to be
in either the susceptible state 0, or the infected state 1, for each week in
the study. We group each inidividual into households of sizes 3, 4 or 5
inidivuals.
"""

# Standard library imports required for the algorithm
from scipy.stats import bernoulli
import math as math
import pandas as pd
import random
import numpy as np
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt
from scipy.stats import beta
import arviz as az

# The parameter alpha denotes the community acquistion infection
alpha = 0.5

# The parameter beta denotes the within-household acquisition infection
beta_variable = 0.1

# The parameter gamma denotes the rate of recovery from the disease infection
gamma = 0.8

# The probability of being infected at the beginning of the observation
mu = 0.5

# The number of individuals in the entire population
number_of_individuals = 200

# The number of discrete time observations made on the infection state of each
# indvidual
number_of_simulations = 10

# Creating an empty list that will contain the infection state of each
# individual at the beginning of the observation
initial_data = []

# Randomly generating the infection state of each inidividual at the beginning
# of the observation
for individual in range(number_of_individuals):
    initial_data.append(bernoulli.rvs(1-mu))

# Creating an empty list that will contain simulated infection state of each
# individual across the entire observation period
simulated_data = []

start = 0

# Grouping each inidividual into a certain household
while start < number_of_individuals:
    # Setting the number of individuals in a particular household
    size = random.randint(3, 8)
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
                    bernoulli.rvs(1-math.exp(-alpha-beta_variable*[
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
    row_names.append('Discrite Time, t=' + str(rows+1))

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

for j in range(number_of_households):
    matrix = np.array(simulated_data[j])

# Initialising the first sample
alpha_0 = 0.9
beta_0 = 0.7
gamma_0 = 0.2
mu = random.uniform(0, 1)

# Creating a variable that counts the number of accepted samples
accepted = 0

# Creating a list that will contain all the accepted proposal states
samples = []

# A list of samples for mu
samples_mu = []

# Craeting a numpy array of the
current_state = np.array([alpha_0, beta_0, gamma_0])

# The total number of generated samples
number_of_samples = 1000000

# The mean of the proposal distribution
mean = np.array([0, 0, 0])

# The covriance matrix of the proposal distribution
covariance_matrix = np.array([[0.1, 0, 0],
                              [0, 0.1, 0],
                              [0, 0, 0.1]])

# Extracting the diagonal elements of the covariance matrix
diag_covariance_matrix = np.diag(covariance_matrix)

# Creating an empty list that will contain all the proposal states
proposal_state = []

while len(proposal_state) < number_of_samples:
    # Generating an array of random numbers
    random_numbers = np.random.normal(mean, np.diag(covariance_matrix))
    # We filter out all the samples containing at least one negative random
    # number
    proposal_state.extend(random_numbers[
        all(number > 0 for number in random_numbers)])

# Reducing the number of proposal states to the number of samples needed
proposal_state = np.array(proposal_state[:number_of_samples])

# Iterating over each individual and household
for h in range(number_of_households):
    for n in range(number_of_individuals):
        samples_mu.append(
            beta((n+1)*(
                h+1)-number_of_individuals+1, number_of_individuals+1))

# Iterating over the number of samples
for sample in range(number_of_samples):

    # Iterating over each household data
    for group in range(number_of_households):

        # Setting the number of rows and columns of each household data
        number_of_rows, number_of_columns = np.array(
            simulated_data[group]).shape

        # Converting the household data list into a numpy array for
        # for computational ease
        household_matrix = np.array(simulated_data[group])

        # Creating a list that will contain the independent conditional
        # probabilities for the likelihood function at the proposal state
        likelihood_probablity_proposal_state = []

        # Creating a list that will contain the independent conditional
        # probabilities for the likelihood function at the current state
        likelihood_probablity_current_state = []

        # Iterating over each individual in the household
        for column in range(number_of_columns):
            # Iterating over each time step in the household data
            for row in range(number_of_rows-1):

                # If an infected individual was initially susceptible at the
                # previous time step
                if household_matrix[row][column] == 0 and household_matrix[
                        row+1][column] == 1:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_proposal_state.append(
                        1-math.exp(
                            -proposal_state[sample][0] - proposal_state[
                                sample][1]*np.sum(
                                    household_matrix[row][:] == 1)))

                # If the susceptible individual was initially susceptible at
                # the previous time step
                elif household_matrix[row][column] == 0 and household_matrix[
                        row+1][column] == 0:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_proposal_state.append(
                        math.exp(
                            -proposal_state[sample][0] - proposal_state[
                                sample][1]*np.sum(
                                    household_matrix[row][:] == 1)))

                # If the susceptible individual was initially infected at the
                # previous time step
                elif household_matrix[row][column] == 1 and household_matrix[
                        row+1][column] == 0:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_proposal_state.append(
                        1-math.exp(-proposal_state[sample][2]))

                # If the infected individual was initially infected at the
                # previous time step
                elif household_matrix[row][column] == 1 and household_matrix[
                        row+1][column] == 1:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_proposal_state.append(
                        math.exp(-proposal_state[sample][2]))

        # Iterating over the columns of the household data
        for column in range(number_of_columns):
            # Iterating over the a certain range dependent on the number of
            # rows of the household data
            for row in range(number_of_rows-1):

                # If an infected individual was initially susceptible at the
                # previous time step
                if household_matrix[row][column] == 0 and household_matrix[
                        row+1][column] == 1:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_current_state.append(
                        1-math.exp(-current_state[0] - current_state[1]*np.sum(
                            household_matrix[row][:] == 1)))

                # If the susceptible individual was initially susceptible at
                # the previous time step
                elif household_matrix[row][column] == 0 and household_matrix[
                        row+1][column] == 0:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_current_state.append(
                        math.exp(-current_state[0] - current_state[1]*np.sum(
                            household_matrix[row][:] == 1)))

                # If the susceptible individual was initially infected at the
                # previous time step
                elif household_matrix[row][column] == 1 and household_matrix[
                        row+1][column] == 0:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_current_state.append(
                        1-math.exp(-current_state[2]))

                # If the infected individual was initially infected at the
                # previous time step
                elif household_matrix[row][column] == 1 and household_matrix[
                        row+1][column] == 1:
                    # Appending the applicable transistion probability into the
                    # list
                    likelihood_probablity_current_state.append(
                        math.exp(-current_state[2]))

    # Computing the acceptance probabiity
    acceptance_probability = min(
        1, (reduce(mul, likelihood_probablity_proposal_state))*math.exp(
            -proposal_state[sample][0])*math.exp(
                -proposal_state[sample][1])*math.exp(
                    -proposal_state[sample][2])/(
                        reduce(
                            mul, likelihood_probablity_current_state)*math.exp(
                                -current_state[0])*math.exp(
                                    -current_state[1])*math.exp(
                                        -current_state[2])))

    # Accepting the proposed state
    if random.uniform(0, 1) < acceptance_probability:
        current_state = proposal_state[sample]
        accepted += 1
        # Store the current state
        samples.append(proposal_state[sample])

# Convert the list of samples to a NumPy array
samples = np.array(samples)

# Print acceptance rate
print(f'Acceptance rate: {accepted / number_of_samples:.2f}')

# Setting the x-axis to be the rates of community aqusition.
samples_alpha = samples[:, 0]

# Plotting the traceplot
az.plot_trace(samples_alpha)

samples_beta = samples[:, 1]

az.plot_trace(samples_beta)

samples_gamma = samples[:, 2]

az.plot_trace(samples_gamma)

az.plot_trace(samples_mu)
