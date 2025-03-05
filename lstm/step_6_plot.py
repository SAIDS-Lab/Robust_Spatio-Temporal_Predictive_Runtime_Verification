import json
import matplotlib.pyplot as plt
import params
import numpy as np
import seaborn as sns

num_bins =30

def main():
    # Plot the histogram of nonconformity scores for the direct method.
    with open(f"experiment_results/{params.num_agents}-agent/direct_nonconformity_list.json", "r") as f:
        direct_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/c_direct.txt", "r") as f:
        c_direct = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/c_tilde_direct.txt", "r") as f:
        c_tilde_direct = float(f.read())

    min_value = min(direct_nonconformity_list)
    max_value = max(direct_nonconformity_list)
    y, x = np.histogram(direct_nonconformity_list, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
    plt.axvline(x=c_direct, color='b', label='$C$')
    plt.axvline(x=c_tilde_direct, color='g', label="$\\tilde{C}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/direct_nonconformity_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the scatter plot of robustnesses for the direct method.
    with open(f"experiment_results/{params.num_agents}-agent/direct_ground_robustnesses.json", "r") as f:
        direct_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/direct_worst_robustnesses_vanilla.json", "r") as f:
        direct_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/direct_worst_robustnesses_robust.json", "r") as f:
        direct_worst_robustnesses_robust = json.load(f)
    x_data = [i for i in range(len(direct_ground_robustnesses))]
    sorted_ground_robustnesses, sorted_worst_robustnesses_vanilla, sorted_worst_robustnesses_robust = zip(
        *sorted(zip(direct_ground_robustnesses, direct_worst_robustnesses_vanilla, direct_worst_robustnesses_robust)))
    dot_sizes = [10 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_ground_robustnesses, s=dot_sizes, marker='x', color='r', linewidths=1.5, label='$\\rho^\psi(X, \\tau_0, l))$')
    plt.scatter(x_data, sorted_worst_robustnesses_vanilla, s=dot_sizes, marker='x', color='b', linewidths=1.5, label='$\\rho^*$ from the Accurate Method')
    plt.scatter(x_data, sorted_worst_robustnesses_robust, s=dot_sizes, marker='x', color='g', linewidths=1.5, label='$\\rho^*$ from the Robust Accurate Method')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\psi(X, \\tau_0, l))$)", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend(fontsize = "large")
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/direct_robustness_scatter" + params.plotting_saving_format)
    plt.show()

    # Load data for the indirect method.
    with open(f"experiment_results/{params.num_agents}-agent/indirect_nonconformity_list.json", "r") as f:
        indirect_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/c_indirect.txt", "r") as f:
        c_indirect = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/c_tilde_indirect.txt", "r") as f:
        c_tilde_indirect = float(f.read())

    # Load data for the hybrid method.
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_nonconformity_list.json", "r") as f:
        hybrid_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/c_hybrid.txt", "r") as f:
        c_hybrid = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/c_tilde_hybrid.txt", "r") as f:
        c_tilde_hybrid = float(f.read())

    # Create bins for the indirect method.
    min_indirect = min(indirect_nonconformity_list)
    max_indirect = max(indirect_nonconformity_list)
    bin_width_indirect = (max_indirect - min_indirect) / num_bins
    bins_indirect = np.arange(min_indirect, max_indirect + bin_width_indirect, bin_width_indirect)
    y_indirect, x_indirect = np.histogram(indirect_nonconformity_list, bins=bins_indirect)

    # Create bins for the hybrid method.
    min_hybrid = min(hybrid_nonconformity_list)
    max_hybrid = max(hybrid_nonconformity_list)
    bin_width_hybrid = (max_hybrid - min_hybrid) / num_bins
    bins_hybrid = np.arange(min_hybrid, max_hybrid + bin_width_hybrid, bin_width_hybrid)
    y_hybrid, x_hybrid = np.histogram(hybrid_nonconformity_list, bins=bins_hybrid)

    # Create a new figure.
    plt.figure()

    # Plot the indirect method histogram (Equation (14)) in blue.
    sns.lineplot(x=x_indirect[:-1], y=y_indirect, color='blue')
    plt.fill_between(x_indirect[:-1], y_indirect, label="Eq. (14) $R^{(i)}$", alpha=0.3, color='blue')

    # Plot the hybrid method histogram (Equation (16)) in orange.
    sns.lineplot(x=x_hybrid[:-1], y=y_hybrid, color='orange')
    plt.fill_between(x_hybrid[:-1], y_hybrid, label="Eq. (16) $R^{(i)}$", alpha=0.3, color='orange')

    # Add vertical lines for the indirect method.
    plt.axvline(x=c_indirect, color='blue', linestyle='--', label="Variant I C")
    plt.axvline(x=c_tilde_indirect, color='blue', linestyle=':', label="Variant I $\\tilde{C}$")

    # Add vertical lines for the hybrid method.
    plt.axvline(x=c_hybrid, color='orange', linestyle='--', label="Variant II C")
    plt.axvline(x=c_tilde_hybrid, color='orange', linestyle=':', label="Variant II $\\tilde{C}$")

    # Formatting the plot.
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()

    # Save and display the plot.
    plt.savefig(f"plots/{params.num_agents}-agent/superimposed_nonconformity_histogram" + params.plotting_saving_format)
    plt.show()


    # Plot the scatter plot of robustnesses for the indirect method.
    with open(f"experiment_results/{params.num_agents}-agent/indirect_ground_robustnesses.json", "r") as f:
        indirect_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_worst_robustnesses_vanilla.json", "r") as f:
        indirect_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_worst_robustnesses_robust.json", "r") as f:
        indirect_worst_robustnesses_robust = json.load(f)
    # Plot the scatter plot of robustnesses for the hybrid method.
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_ground_robustnesses.json", "r") as f:
        hybrid_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_worst_robustnesses_vanilla.json", "r") as f:
        hybrid_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_worst_robustnesses_robust.json", "r") as f:
        hybrid_worst_robustnesses_robust = json.load(f)
    x_data = [i for i in range(len(hybrid_ground_robustnesses))]
    sorted_ground_indirect_robustnesses, sorted_worst_indirect_robustnesses_vanilla, sorted_worst_indirect_robustnesses_robust, sorted_worst_hybrid_robustnesses_vanilla, sorted_worst_hybrid_robustnesses_robust = zip(
        *sorted(zip(indirect_ground_robustnesses, indirect_worst_robustnesses_vanilla,
                    indirect_worst_robustnesses_robust, hybrid_worst_robustnesses_vanilla, hybrid_worst_robustnesses_robust)))
    dot_sizes = [10 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_ground_indirect_robustnesses, s=dot_sizes, marker='x',linewidths=1.5, color='r', label='$\\rho^\psi(X, \\tau_0, l))$')
    plt.scatter(x_data, sorted_worst_indirect_robustnesses_vanilla, s=dot_sizes, marker='x',linewidths=1.5, color='b',
                label='$\\rho^*$ (Interpretable Method Variant I)')
    plt.scatter(x_data, sorted_worst_indirect_robustnesses_robust, s=dot_sizes, marker='x',linewidths=1.5, color='g',
                label='$\\rho^*$ (Robust Interpretable Method Variant I)')
    plt.scatter(x_data, sorted_worst_hybrid_robustnesses_vanilla, s = dot_sizes, color = 'm', marker='x',linewidths=1.5, label = "$\\rho^*$ (Interpretable Method Variant II)")
    plt.scatter(x_data, sorted_worst_hybrid_robustnesses_robust, s = dot_sizes, color = 'k', marker='x',linewidths=1.5, label = "$\\rho^*$ (Robust Interpretable Method Variant II)")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\psi(X, \\tau_0, l))$)", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend(fontsize = "large")
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/hybrid_robustness_scatter" + params.plotting_saving_format)
    plt.show()

    # Plot the coverage histograms for the direct methods.
    with open(f"experiment_results/{params.num_agents}-agent/direct_vanilla_coverages.json", "r") as f:
        direct_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/direct_robust_coverages.json", "r") as f:
        direct_robust_coverages = json.load(f)
    min_value = min(np.concatenate([direct_vanilla_coverages, direct_robust_coverages]))
    max_value = max(np.concatenate([direct_vanilla_coverages, direct_robust_coverages]))
    y, x = np.histogram(direct_vanilla_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Accurate Runtime Verification Method")
    y, x = np.histogram(direct_robust_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Robust Accurate Runtime Verification Method")
    plt.axvline(x = 1 - params.delta, label = "Expected Coverage Rate", color = "k", linestyle = "--")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.legend(fontsize= "large")
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/direct_coverage_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the coverage histogram for the indirect methods.
    with open(f"experiment_results/{params.num_agents}-agent/indirect_vanilla_coverages.json", "r") as f:
        indirect_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/indirect_robust_coverages.json", "r") as f:
        indirect_robust_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_vanilla_coverages.json", "r") as f:
        hybrid_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/hybrid_robust_coverages.json", "r") as f:
        hybrid_robust_coverages = json.load(f)
    min_value = min(np.concatenate([indirect_vanilla_coverages, indirect_robust_coverages, hybrid_vanilla_coverages, hybrid_robust_coverages])) - 0.05
    max_value = max(np.concatenate([indirect_vanilla_coverages, indirect_robust_coverages, hybrid_vanilla_coverages, hybrid_robust_coverages])) + 0.05
    y, x = np.histogram(indirect_vanilla_coverages, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Interpretable Method Variant I")
    y, x = np.histogram(indirect_robust_coverages, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Robust Interpretable Method Variant I")
    y, x = np.histogram(hybrid_vanilla_coverages, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Interpretable Method Variant II")
    y, x = np.histogram(hybrid_robust_coverages, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label = "Robust Interpretable Method Variant II")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Coverage", fontsize = params.font_size)
    plt.ylabel("Frequency", fontsize = params.font_size)
    plt.legend(fontsize="large")
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/indirect_coverage_histogram" + params.plotting_saving_format)
    plt.show()

    # Analyze the timings.
    with open(f"experiment_results/{params.num_agents}-agent/direct_prediction_region_calculation_times.json", "r") as f:
        direct_prediction_region_calculation_times = json.load(f)
        direct_training_times = [float(time) for time in direct_prediction_region_calculation_times]
        average_direct_training_time = np.average(direct_training_times)
        plt.hist(direct_training_times, bins = 20)
        plt.axvline(average_direct_training_time, label = f"Average Time: {average_direct_training_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/direct_prediction_region_calculation_times" + params.plotting_saving_format)
        plt.show()

    with open(f"experiment_results/{params.num_agents}-agent/direct_test_times.json", "r") as f:
        direct_test_times = json.load(f)
        direct_test_times = [float(time) for time in direct_test_times]
        average_direct_test_time = np.average(direct_test_times)
        plt.hist(direct_test_times, bins = 20)
        plt.axvline(average_direct_test_time, label = f"Average Time: {average_direct_test_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/direct_test_times" + params.plotting_saving_format)
        plt.show()

    with open(f"experiment_results/{params.num_agents}-agent/indirect_prediction_region_calculation_times.json", "r") as f:
        indirect_prediction_region_calculation_times = json.load(f)
        indirect_training_times = [float(time) for time in indirect_prediction_region_calculation_times]
        average_indirect_training_time = np.average(indirect_training_times)
        plt.hist(indirect_training_times, bins = 20)
        plt.axvline(average_indirect_training_time, label = f"Average Time: {average_indirect_training_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/indirect_prediction_region_calculation_times" + params.plotting_saving_format)
        plt.show()

    with open(f"experiment_results/{params.num_agents}-agent/indirect_test_times.json", "r") as f:
        indirect_test_times = json.load(f)
        indirect_test_times = [float(time) for time in indirect_test_times]
        average_indirect_test_time = np.average(indirect_test_times)
        plt.hist(indirect_test_times, bins = 20)
        plt.axvline(average_indirect_test_time, label = f"Average Time: {average_indirect_test_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/indirect_test_times" + params.plotting_saving_format)
        plt.show()

    with open(f"experiment_results/{params.num_agents}-agent/hybrid_prediction_region_calculation_times.json", "r") as f:
        hybrid_prediction_region_calculation_times = json.load(f)
        hybrid_training_times = [float(time) for time in hybrid_prediction_region_calculation_times]
        average_hybrid_training_time = np.average(hybrid_training_times)
        plt.hist(hybrid_training_times, bins = 20)
        plt.axvline(average_hybrid_training_time, label = f"Average Time: {average_hybrid_training_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/hybrid_prediction_region_calculation_times" + params.plotting_saving_format)
        plt.show()

    with open(f"experiment_results/{params.num_agents}-agent/hybrid_test_times.json", "r") as f:
        hybrid_test_times = json.load(f)
        hybrid_test_times = [float(time) for time in hybrid_test_times]
        average_hybrid_test_time = np.average(hybrid_test_times)
        plt.hist(hybrid_test_times, bins = 20)
        plt.axvline(average_hybrid_test_time, label = f"Average Time: {average_hybrid_test_time}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.legend(fontsize=params.legend_size)
        plt.tight_layout()
        plt.savefig(f"plots/{params.num_agents}-agent/hybrid_test_times" + params.plotting_saving_format)
        plt.show()


if __name__ == '__main__':
    main()