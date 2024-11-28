data_alpha = [samples_alpha_non_adaptive, samples_alpha_adaptive]
# Create a figure and axis
fig, ax = plt.subplots()
# Create the boxplot
box = ax.boxplot(
     data_alpha, patch_artist=True, widths=0.6, notch=True,
     boxprops=dict(facecolor='lightgreen', linewidth=2),
     whiskerprops=dict(color='black', linewidth=2),
     capprops=dict(color='black', linewidth=2),
     flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
     medianprops=dict(color='black', linewidth=1.5))

 # Set labels for x-axis
 ax.set_xticklabels(['1', '2'])

  # Define colours for each Algorithm
  colors = ['lightblue', 'lightgreen']

   # Apply colours to each boxplot
   for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Creating custom labels
    labels = ['Metropolis within Gibbs', 'Adaptive Metropolis within Gibbs', ]

    # The line representing the expected value
    plt.axhline(alpha, color="red", linestyle="--",
                label=f"Expected Value $\\alpha$: {alpha:.2f}")

    # Setting the x-axis label
    plt.xlabel("Algorithm")

    # Settingg the y-axis label
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Community aquisition infection rate $\\alpha$",
              fontsize=16, fontweight='bold')

    # Setting the legend
    ax.legend([box['boxes'][i] for i in range(len(colors))], labels,
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid line visibility
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # The generated MCMC samples for the with-in household aquisition infection
    # rate

    data_beta = [samples_beta_non_adaptive, samples_beta_adaptive]
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Create the boxplot
    box = ax.boxplot(
        data_beta, patch_artist=True, widths=0.6, notch=True,
        boxprops=dict(facecolor='lightgreen', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
        medianprops=dict(color='black', linewidth=1.5))

    # Set labels for x-axis
    ax.set_xticklabels(['1', '2'])

    # Define colours for each Algorithm
    colors = ['lightblue', 'lightgreen']

    # Apply colours to each boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Creating custom labels
    labels = ['Metropolis within Gibbs', 'Adaptive Metropolis within Gibbs', ]

    # The line representing the expected value
    plt.axhline(beta, color="red", linestyle="--",
                label=f"Expected Value $\\beta$: {beta:.2f}")

    # Setting the x-axis label
    plt.xlabel("Algorithm")

    # Settingg the y-axis label
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Within household infection rate $\\beta$",
              fontsize=16, fontweight='bold')

    # Setting the legend
    ax.legend([box['boxes'][i] for i in range(len(colors))], labels,
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid line visibility
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    data_gamma = [samples_gamma_non_adaptive, samples_gamma_adaptive]
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Create the boxplot
    box = ax.boxplot(
        data_gamma, patch_artist=True, widths=0.6, notch=True,
        boxprops=dict(facecolor='lightgreen', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
        medianprops=dict(color='black', linewidth=1.5))

    # Set labels for x-axis
    ax.set_xticklabels(['1', '2'])

    # Define colours for each Algorithm
    colors = ['lightblue', 'lightgreen']

    # Apply colours to each boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Creating custom labels
    labels = ['Metropolis within Gibbs', 'Adaptive Metropolis within Gibbs', ]

    # The line representing the expected value
    plt.axhline(gamma, color="red", linestyle="--",
                label=f"Expected Value $\\gamma$: {gamma:.2f}")

    # Setting the x-axis label
    plt.xlabel("Algorithm")

    # Settingg the y-axis label
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("The recovery rate $\\gamma$", fontsize=16, fontweight='bold')

    # Setting the legend
    ax.legend([box['boxes'][i] for i in range(len(colors))], labels,
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Setting the grid line visibility
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()

    # Setting the figure size
    plt.figure(figsize=(10, 6))

    # Creating the boxplot
    plt.boxplot(
        samples_mu, patch_artist=True, widths=0.6, notch=True,
        boxprops=dict(facecolor='gray', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
        medianprops=dict(color='black', linewidth=1.5))

    # The line representing the expected value
    plt.axhline(mu, color="red", linestyle="--",
                label=f"Expected Value: {mu:.2f}")
    # Labelling the x-axis
    plt.xlabel("Iteration")

    # Labelling the y-axis
    plt.ylabel("Parameter Value")

    # Setting the title
    plt.title("Probability of being infceted at the first observation week $\\mu$",
              fontsize=16,  fontweight='bold')

    # The line representing the expected value
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Settin the grid lines
    plt.grid(alpha=0.3)

    # Show the plot
    plt.show()
