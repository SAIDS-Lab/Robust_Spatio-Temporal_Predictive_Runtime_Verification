import json
import matplotlib.pyplot as plt
import params
import numpy as np
import seaborn as sns

num_bins = 30
def main():
    # Plot the histogram of nonconformity scores for the direct method.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_nonconformity_list.json", "r") as f:
        direct_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_direct.txt", "r") as f:
        c_direct = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_direct.txt", "r") as f:
        c_tilde_direct = float(f.read())
    min_value = min(direct_nonconformity_list)
    max_value = max(direct_nonconformity_list)
    y, x = np.histogram(direct_nonconformity_list,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
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
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_direct_nonconformity_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the coverage histograms for the direct methods.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_vanilla_coverages.json", "r") as f:
        direct_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_robust_coverages.json", "r") as f:
        direct_robust_coverages = json.load(f)
    min_value = min(np.concatenate([direct_vanilla_coverages, direct_robust_coverages]))
    max_value = max(np.concatenate([direct_vanilla_coverages, direct_robust_coverages]))
    y, x = np.histogram(direct_vanilla_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Accurate Runtime Verification Method")
    y, x = np.histogram(direct_robust_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Robust Accurate Runtime Verification Method")
    plt.axvline(x=1 - params.delta, label="Expected Coverage Rate", color="k", linestyle="--")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Coverage", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_direct_coverage_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the scatter plot of robustnesses for the direct method.
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_ground_robustnesses.json", "r") as f:
        direct_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_worst_robustnesses_vanilla.json", "r") as f:
        direct_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/direct_worst_robustnesses_robust.json", "r") as f:
        direct_worst_robustnesses_robust = json.load(f)
    x_data = [i for i in range(len(direct_ground_robustnesses))]
    sorted_ground_robustnesses, sorted_worst_robustnesses_vanilla, sorted_worst_robustnesses_robust = zip(
        *sorted(zip(direct_ground_robustnesses, direct_worst_robustnesses_vanilla, direct_worst_robustnesses_robust)))
    dot_sizes = [5 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_ground_robustnesses, s=dot_sizes, color='r', label='$\\rho^\phi(X, \\tau_0))$')
    plt.scatter(x_data, sorted_worst_robustnesses_vanilla, s=dot_sizes, color='b',
                label='$\\rho^*$ from the Accurate Method')
    plt.scatter(x_data, sorted_worst_robustnesses_robust, s=dot_sizes, color='g',
                label='$\\rho^*$ from the Robust Accurate Method')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\phi(X, \\tau_0)$)", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_direct_robustness_scatter" + params.plotting_saving_format)
    plt.show()

    # Plot the histogram of nonconformity scores for the indirect method.
    # Load indirect alphas.
    with open(f'alphas/{params.num_agents}-agent/STL/alphas_indirect.json', 'r') as f:
        alphas_indirect = json.load(f)
        new_alphas_indirect = dict()
        for tau in alphas_indirect:
            new_alphas_indirect[int(tau)] = alphas_indirect[tau]
        alphas_indirect = new_alphas_indirect
    with open(f"experiment_results/{params.num_agents}-agent/STL/old_indirect_nonconformity_list.json", "r") as f:
        indirect_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_indirect.txt", "r") as f:
        c_indirect = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_indirect.txt", "r") as f:
        c_tilde_indirect = float(f.read())
    # Compute C.
    indirect_nonconformity_list.sort()
    indirect_nonconformity_list.append(float("inf"))
    p = int(np.ceil((len(indirect_nonconformity_list)) * (1 - params.delta / (params.T - params.current_time))))
    c = indirect_nonconformity_list[p - 1]
    min_value = min(indirect_nonconformity_list[:-1])
    max_value = max(indirect_nonconformity_list[:-1])
    y, x = np.histogram(indirect_nonconformity_list[:-1],
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                        (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
    plt.axvline(x=c_indirect * alphas_indirect[params.indirect_illustration_variant_1_tau], color='b', label='$C\\alpha_\\tau$')
    plt.axvline(x=c_tilde_indirect * alphas_indirect[params.indirect_illustration_variant_1_tau], color='g', label="$\\tilde{C}\\alpha_\\tau$")
    plt.axvline(x=c, color = 'r', label = '$C$ from baseline')
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_indirect_nonconformity_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the histogram of nonconformity scores for the hybrid method.
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_nonconformity_list.json", "r") as f:
        hybrid_nonconformity_list = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_hybrid.txt", "r") as f:
        c_hybrid = float(f.read())
    with open(f"experiment_results/{params.num_agents}-agent/STL/c_tilde_hybrid.txt", "r") as f:
        c_tilde_hybrid = float(f.read())
    min_value = min(hybrid_nonconformity_list)
    max_value = max(hybrid_nonconformity_list)
    y, x = np.histogram(hybrid_nonconformity_list,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
    plt.axvline(x=c_hybrid, color='b', label='$C$')
    plt.axvline(x=c_tilde_hybrid, color='g', label="$\\tilde{C}$")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Nonconformity Score", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize = params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_hybrid_nonconformity_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the coverage histogram for the indirect methods.
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_vanilla_coverages.json", "r") as f:
        indirect_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_robust_coverages.json", "r") as f:
        indirect_robust_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_vanilla_coverages.json", "r") as f:
        hybrid_vanilla_coverages = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_robust_coverages.json", "r") as f:
        hybrid_robust_coverages = json.load(f)
    min_value = min(np.concatenate([indirect_vanilla_coverages, indirect_robust_coverages, hybrid_vanilla_coverages,
                                    hybrid_robust_coverages])) - 0.05
    max_value = max(np.concatenate([indirect_vanilla_coverages, indirect_robust_coverages, hybrid_vanilla_coverages,
                                    hybrid_robust_coverages])) + 0.05
    y, x = np.histogram(indirect_vanilla_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Interpretable Method Variant I")
    y, x = np.histogram(indirect_robust_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Robust Interpretable Method Variant I")
    y, x = np.histogram(hybrid_vanilla_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Interpretable Method Variant II")
    y, x = np.histogram(hybrid_robust_coverages,
                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                       (max_value - min_value) / num_bins))
    sns.lineplot(x=x[:-1], y=y)
    plt.fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label="Robust Interpretable Method Variant II")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Coverage", fontsize=params.font_size)
    plt.ylabel("Frequency", fontsize=params.font_size)
    plt.legend(fontsize=params.legend_size)
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_indirect_coverage_histogram" + params.plotting_saving_format)
    plt.show()

    # Plot the scatter plot of robustnesses for the indirect method.
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_ground_robustnesses.json", "r") as f:
        indirect_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_worst_robustnesses_vanilla.json", "r") as f:
        indirect_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/indirect_worst_robustnesses_robust.json", "r") as f:
        indirect_worst_robustnesses_robust = json.load(f)
    # Plot the scatter plot of robustnesses for the hybrid method.
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_ground_robustnesses.json", "r") as f:
        hybrid_ground_robustnesses = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_worst_robustnesses_vanilla.json", "r") as f:
        hybrid_worst_robustnesses_vanilla = json.load(f)
    with open(f"experiment_results/{params.num_agents}-agent/STL/hybrid_worst_robustnesses_robust.json", "r") as f:
        hybrid_worst_robustnesses_robust = json.load(f)
    x_data = [i for i in range(len(hybrid_ground_robustnesses))]
    sorted_ground_indirect_robustnesses, sorted_worst_indirect_robustnesses_vanilla, sorted_worst_indirect_robustnesses_robust, sorted_worst_hybrid_robustnesses_vanilla, sorted_worst_hybrid_robustnesses_robust = zip(
        *sorted(zip(indirect_ground_robustnesses, indirect_worst_robustnesses_vanilla,
                    indirect_worst_robustnesses_robust, hybrid_worst_robustnesses_vanilla,
                    hybrid_worst_robustnesses_robust)))
    dot_sizes = [5 for i in range(len(x_data))]
    plt.scatter(x_data, sorted_ground_indirect_robustnesses, s=dot_sizes, color='r',
                label='$\\rho^\phi(X, \\tau_0))$')
    plt.scatter(x_data, sorted_worst_indirect_robustnesses_vanilla, s=dot_sizes, color='b',
                label='$\\rho^*$ (Interpretable Method Variant I)')
    plt.scatter(x_data, sorted_worst_indirect_robustnesses_robust, s=dot_sizes, color='g',
                label='$\\rho^*$ (Robust Interpretable Method Variant I)')
    plt.scatter(x_data, sorted_worst_hybrid_robustnesses_vanilla, s=dot_sizes, color='m',
                label="$\\rho^*$ (Interpretable Method Variant II)")
    plt.scatter(x_data, sorted_worst_hybrid_robustnesses_robust, s=dot_sizes, color='k',
                label="$\\rho^*$ (Robust Interpretable Method Variant II)")
    plt.tick_params("x", labelsize=params.label_size)
    plt.tick_params("y", labelsize=params.label_size)
    plt.xlabel("Sample (Sorted on $\\rho^\phi(X, \\tau_0))$)", fontsize=params.font_size)
    plt.ylabel("Robust Semantics Value", fontsize=params.font_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{params.num_agents}-agent/STL/stl_hybrid_robustness_scatter" + params.plotting_saving_format)
    plt.show()


if __name__ == '__main__':
    main()