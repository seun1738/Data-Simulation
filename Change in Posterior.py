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
import scipy.stats as st
import statistics

for j in range(30):
    # The community acquistion infection rate
    alpha = 0.1

    # The within-household acquisition infection rate
    beta = 0.2

    # The  rate of recovery from the disease infection
    gamma = 0.2

    # The probability of being infected at the first week of the observation period
    mu = 0.2

    # The number of individuals in the population
    number_of_individuals = 200

    # The number of discrete time observations made on the infection state of each
    # indvidual
    observation_period = 100

    # Creating an empty list that will contain the infection state of each
    # individual at the first week of the observation period
    initial_data = []

    # Initialising the random generator
    random.seed(42)

    # Generating the infection state of each inidividual at the first week of the
    # observation
    for individual in range(number_of_individuals):
        initial_data.append(bernoulli.rvs(mu))

    # Creating an empty list that will contain simulated infection state of each
    # individual across the entire observation period
    simulated_data = []

    # Setting a counter
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

    # Simulating the infectious disease dynamics in each household throughout the
    # observation period
    for household in range(number_of_households):
        index = 0
        # Simulating the infectious disease dynamics for each individual in the
        # household at all observation time points
        for simulations in range(observation_period):

            # Creating an empty list containing the infection state data of
            # each individual at a certain week
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
                        bernoulli.rvs(math.exp(-gamma)))
                index += 1
            # Appending the infection state of all inidividuals at a particular
            # observation week into the population data
            for state in infection_state_at_certain_time_step:
                simulated_data[household].append(state)

    # Creating column names for each hosuehold pandas dataframe
    column_names = []
    for columns in range(number_of_individuals):
        column_names.append('Individual ' + str(columns+1))

    # Creating row names for each household pandas dataframe
    row_names = []
    for rows in range(observation_period+1):
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

    # Initialising the first sample
    alpha_0 = 1.5
    beta_0 = 1.5
    gamma_0 = 1.5

    # Creating a variable that counts the number of accepted samples
    accepted = 0

    # A list of all proposed samples for the probability an individual is infected
    # at the first week of the obervation
    samples_mu_pone = []

    # Craeting a numpy array of the first sample current state
    current_state = np.array([alpha_0, beta_0, gamma_0])

    # The total number of generated samples
    number_of_samples = 50000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # The covriance matrix of the proposal distribution
    covariance_matrix = np.array([[0.1, 0, 0],
                                  [0, 0.1, 0],
                                  [0, 0, 0.1]])

    # Setting the number of infected individuals at the first week of the
    # observation period
    number_of_infected_individuals_at_week_one = initial_data.count(1)
    number_of_susceptible_individuals_at_week_one = number_of_individuals -\
        initial_data.count(1)

    # Setting the x-parameter of the beta distribution to be in terms of the number
    # of infected individuals at the first week of the observation period
    x = number_of_infected_individuals_at_week_one + 1

    # Setting the y-parameter of the beta distribution to be in terms of the number
    # of susceptible individuals at the first week of the observation period
    y = number_of_susceptible_individuals_at_week_one + 1

    # Creating a numpy-array of the generated MCMC samples
    samples_mu_pone = np.random.beta(x, y, 50000)

    # The dimension of our samples
    dimension = 3

    # The burn-in phase of our MCMC chain
    burn_in = 10000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # Creating a variable that counts the number of accepted samples
    samples_adaptive_pone = []

    # The total number of generated samples
    number_of_samples = 50000

    # Iterating over the number of MCMC samples we want ot generate during the
    # burn-in phase
    for sample in range(burn_in):

        # Generating a proposal sample
        proposal_state = current_state + \
            np.random.multivariate_normal(
                mean, (0.1**2)*np.identity(dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Store the current state
            samples_adaptive_pone.append(proposal_state)

    # Setting the value of epsilon
    epsilon = 0.05

    # Iterating over the number of samples
    for sample in range(burn_in-1, number_of_samples):
        covariance_matrix = np.cov(samples_adaptive_pone, rowvar=False)

        proposal_state = current_state + (1-epsilon)*np.random.multivariate_normal(
            mean, (2.38**2)*covariance_matrix / dimension) + \
            epsilon*np.random.multivariate_normal(mean, (0.1**2)*np.identity(
                dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Storing the current state
            samples_adaptive_pone.append(proposal_state)

    # Convert the list of samples after the burn-in phase to a NumPy array
    samples_after_burn_in_pone = np.array(samples_adaptive_pone[burn_in:])

    samples_alpha_adaptive_pone = samples_after_burn_in_pone[:, 0]

    # Initialising the first sample
    alpha_0 = 1
    beta_0 = 1
    gamma_0 = 1

    # Creating a variable that counts the number of accepted samples
    accepted = 0

    # A list of all proposed samples for the probability an individual is infected
    # at the first week of the obervation
    samples_mu_ptwo = []

    # Craeting a numpy array of the first sample current state
    current_state = np.array([alpha_0, beta_0, gamma_0])

    # The total number of generated samples
    number_of_samples = 50000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # The covriance matrix of the proposal distribution
    covariance_matrix = np.array([[0.1, 0, 0],
                                  [0, 0.1, 0],
                                  [0, 0, 0.1]])

    # Setting the number of infected individuals at the first week of the
    # observation period
    number_of_infected_individuals_at_week_one = initial_data.count(1)
    number_of_susceptible_individuals_at_week_one = number_of_individuals -\
        initial_data.count(1)

    # Setting the x-parameter of the beta distribution to be in terms of the number
    # of infected individuals at the first week of the observation period
    x = number_of_infected_individuals_at_week_one + 1

    # Setting the y-parameter of the beta distribution to be in terms of the number
    # of susceptible individuals at the first week of the observation period
    y = number_of_susceptible_individuals_at_week_one + 1

    # Creating a numpy-array of the generated MCMC samples
    samples_mu_ptwo = np.random.beta(x, y, 50000)

    # The dimension of our samples
    dimension = 3

    # The burn-in phase of our MCMC chain
    burn_in = 10000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # Creating a variable that counts the number of accepted samples
    samples_adaptive_ptwo = []

    # The total number of generated samples
    number_of_samples = 50000

    # Iterating over the number of MCMC samples we want ot generate during the
    # burn-in phase
    for sample in range(burn_in):

        # Generating a proposal sample
        proposal_state = current_state + \
            np.random.multivariate_normal(
                mean, (0.1**2)*np.identity(dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Store the current state
            samples_adaptive_ptwo.append(proposal_state)

    # Setting the value of epsilon
    epsilon = 0.05

    # Iterating over the number of samples
    for sample in range(burn_in-1, number_of_samples):
        covariance_matrix = np.cov(samples_adaptive_ptwo, rowvar=False)

        proposal_state = current_state + (1-epsilon)*np.random.multivariate_normal(
            mean, (2.38**2)*covariance_matrix / dimension) + \
            epsilon*np.random.multivariate_normal(mean, (0.1**2)*np.identity(
                dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Storing the current state
            samples_adaptive_ptwo.append(proposal_state)

    # Convert the list of samples after the burn-in phase to a NumPy array
    samples_after_burn_in_ptwo = np.array(samples_adaptive_ptwo[burn_in:])

    # Initialising the first sample
    alpha_0 = 0.5
    beta_0 = 0.5
    gamma_0 = 0.5

    # Creating a variable that counts the number of accepted samples
    accepted = 0

    # A list of all proposed samples for the probability an individual is infected
    # at the first week of the obervation
    samples_mu_pthree = []

    # Craeting a numpy array of the first sample current state
    current_state = np.array([alpha_0, beta_0, gamma_0])

    # The total number of generated samples
    number_of_samples = 50000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # The covriance matrix of the proposal distribution
    covariance_matrix = np.array([[0.1, 0, 0],
                                  [0, 0.1, 0],
                                  [0, 0, 0.1]])

    # Setting the number of infected individuals at the first week of the
    # observation period
    number_of_infected_individuals_at_week_one = initial_data.count(1)
    number_of_susceptible_individuals_at_week_one = number_of_individuals -\
        initial_data.count(1)

    # Setting the x-parameter of the beta distribution to be in terms of the number
    # of infected individuals at the first week of the observation period
    x = number_of_infected_individuals_at_week_one + 1

    # Setting the y-parameter of the beta distribution to be in terms of the number
    # of susceptible individuals at the first week of the observation period
    y = number_of_susceptible_individuals_at_week_one + 1

    # Creating a numpy-array of the generated MCMC samples
    samples_mu_pthree = np.random.beta(x, y, 50000)

    # The dimension of our samples
    dimension = 3

    # The burn-in phase of our MCMC chain
    burn_in = 10000

    # The mean of the proposal distribution
    mean = np.array([0, 0, 0])

    # Creating a variable that counts the number of accepted samples
    samples_adaptive_pthree = []

    # The total number of generated samples
    number_of_samples = 50000

    # Iterating over the number of MCMC samples we want ot generate during the
    # burn-in phase
    for sample in range(burn_in):

        # Generating a proposal sample
        proposal_state = current_state + \
            np.random.multivariate_normal(
                mean, (0.1**2)*np.identity(dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Store the current state
            samples_adaptive_pthree.append(proposal_state)

    # Setting the value of epsilon
    epsilon = 0.05

    # Iterating over the number of samples
    for sample in range(burn_in-1, number_of_samples):
        covariance_matrix = np.cov(samples_adaptive_pthree, rowvar=False)

        proposal_state = current_state + (1-epsilon)*np.random.multivariate_normal(
            mean, (2.38**2)*covariance_matrix / dimension) + \
            epsilon*np.random.multivariate_normal(mean, (0.1**2)*np.identity(
                dimension)/dimension)

        # If all the values in the proposed sample are positive
        if all(number > 0 for number in proposal_state):

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
                    # Iterating over each observation week in the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(
                                    -proposal_state[0] - proposal_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                1-math.exp(-proposal_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_proposal_state.append(
                                math.exp(-proposal_state[2]))

                # Iterating over the columns of the household data
                for column in range(number_of_columns):
                    # Iterating over the a certain range dependent on the number of
                    # rows of the household data
                    for row in range(number_of_rows-1):

                        # If an infected individual was initially susceptible at
                        # the previous time step
                        if household_matrix[row][column] == 0 and household_matrix[
                                row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially susceptible
                        # at the previous time step
                        elif household_matrix[
                                row][column] == 0 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(
                                    -current_state[0] - current_state[1]*np.sum(
                                        household_matrix[row][:] == 1)))

                        # If the susceptible individual was initially infected at
                        # the previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 0:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                1-math.exp(-current_state[2]))

                        # If the infected individual was initially infected at the
                        # previous time step
                        elif household_matrix[
                                row][column] == 1 and household_matrix[
                                    row+1][column] == 1:
                            # Appending the applicable transistion probability into
                            # the list
                            likelihood_probablity_current_state.append(
                                math.exp(-current_state[2]))

            # Computing the acceptance probabiity
            acceptance_probability = min(
                1, (reduce(
                    mul, likelihood_probablity_proposal_state)*np.exp(
                        -proposal_state[0])*np.exp(-proposal_state[1])*np.exp(
                            -proposal_state[2]))/(reduce(
                                mul, likelihood_probablity_current_state)*np.exp(
                                    -current_state[0])*np.exp(
                                        -current_state[1])*np.exp(
                                            -current_state[2])))

            # Accepting the proposed state
            if random.uniform(0, 1) < acceptance_probability:
                current_state = proposal_state
                accepted += 1

            # Storing the current state
            samples_adaptive_pthree.append(proposal_state)

    # Convert the list of samples after the burn-in phase to a NumPy array

    samples_adaptive_pone = np.array(samples_adaptive_pone)
    samples_adaptive_ptwo = np.array(samples_adaptive_ptwo)
    samples_adaptive_pthree = np.array(samples_adaptive_pthree)

    samples_alpha_pone = samples_adaptive_pone[:, 0]
    samples_alpha_ptwo = samples_adaptive_ptwo[:, 0]
    samples_alpha_pthree = samples_adaptive_pthree[:, 0]

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_alpha_pone, label="Trace",
             alpha=0.7, color='dodgerblue')
    plt.plot(samples_alpha_ptwo, label="Trace",
             alpha=0.7, color='orange')
    plt.plot(samples_alpha_pone, label="Trace",
             alpha=0.7, color='forestgreen')

    # The line representing the expected value
    plt.axhline(alpha, color="red", linestyle="--",
                label=f"Expected Value of $\\alpha$: {alpha:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\alpha$", fontsize=16,  fontweight='bold')

    # Setting the legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)

    # Setting the grid lines
    plt.grid(alpha=0.3)

    # Sisplaying the plot
    plt.show()

    samples_beta_non_adaptive = samples_non_adaptive[:, 1]
    samples_beta_adaptive = samples_after_burn_in[:, 1]

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_bet, label="Trace",
             color='mediumseagreen', alpha=0.7)

    # The line representing the expected value
    plt.axhline(beta, color="red", linestyle="--",
                label=f"Expected Value of $\\beta$: {beta:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\beta$(non-apative)",
              fontsize=16,  fontweight='bold')

    # Setting the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid lines
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # The generated MCMC samples for the recovery rate
    samples_gamma_non_adaptive = samples_non_adaptive[:, 2]
    samples_gamma_adaptive = samples_after_burn_in[:, 2]

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_gamma_non_adaptive, label="Trace",
             color='mediumseagreen', alpha=0.7)

    # The line representing the expected value
    plt.axhline(gamma, color="red", linestyle="--",
                label=f"Expected Value of $\\gamma$: {gamma:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\gamma$(apative)", fontsize=16,  fontweight='bold')

    # The line representing the expected value
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Settin the grid lines
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # Creating the traceplot
    plt.plot(samples_mu, label="Trace", alpha=0.7, color='mediumseagreen')

    # The line representing the expected value
    plt.axhline(mu, color="red", linestyle="--",
                label=f"Expected Value: {mu:.2f}")

    # Setting the x-axis label
    plt.xlabel("Iteration")

    # Settingg the y-axis label
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\mu$", fontsize=16, fontweight='bold')

    # Setting the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid line visibility
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_alpha_adaptive, label="Trace", alpha=0.7)

    # The line representing the expected value
    plt.axhline(alpha, color="red", linestyle="--",
                label=f"Expected Value of $\\alpha$: {alpha:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\alpha$(adaptive)", fontsize=16,  fontweight='bold')

    # Setting the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid lines
    plt.grid(alpha=0.3)

    # Displaying the plot
    plt.show()

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_beta_adaptive, label="Trace", alpha=0.7)

    # The line representing the expected value
    plt.axhline(beta, color="red", linestyle="--",
                label=f"Expected Value of $\\beta$: {beta:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\beta$", fontsize=16,  fontweight='bold')

    # Setting the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid lines
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the traceplot
    plt.plot(samples_gamma_adaptive, label="Trace", alpha=0.7)

    # The line representing the expected value
    plt.axhline(gamma, color="red", linestyle="--",
                label=f"Expected Value of $\\gamma$: {gamma:.2f}")

    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\gamma$(apative)", fontsize=16,  fontweight='bold')

    # The line representing the expected value
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Settin the grid lines
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # Creating the traceplot
    plt.plot(samples_mu, label="Trace", alpha=0.7)

    # The line representing the expected value
    plt.axhline(mu, color="red", linestyle="--",
                label=f"Expected Value: {mu:.2f}")

    # Setting the x-axis label
    plt.xlabel("Iteration")

    # Settingg the y-axis label
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Trace of $\\mu$", fontsize=16, fontweight='bold')

    # Setting the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid line visibility
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()
