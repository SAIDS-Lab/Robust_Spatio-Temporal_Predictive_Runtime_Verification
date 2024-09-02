import json
import matplotlib.pyplot as plt
import params
import numpy as np


def main():
    # Load the timings.
    colormap = dict()
    colormap[5] = "r"
    colormap[7] = "b"
    colormap[10] = "g"

    direct_calibration_times = dict()
    indirect_calibration_times = dict()
    hybrid_calibration_times = dict()
    direct_test_times = dict()
    indirect_test_times = dict()
    hybrid_test_times = dict()
    for N in [5, 7, 10]:
        with open(f"experiment_results/{N}-agent/direct_prediction_region_calculation_times.json",
                  "r") as f:
            direct_prediction_region_calculation_times = json.load(f)
            direct_training_times = [float(time) for time in direct_prediction_region_calculation_times]
            direct_calibration_times[N] = direct_training_times
        with open(f"experiment_results/{N}-agent/indirect_prediction_region_calculation_times.json",
                  "r") as f:
            indirect_prediction_region_calculation_times = json.load(f)
            indirect_training_times = [float(time) for time in indirect_prediction_region_calculation_times]
            indirect_calibration_times[N] = indirect_training_times
        with open(f"experiment_results/{N}-agent/hybrid_prediction_region_calculation_times.json",
                  "r") as f:
            hybrid_prediction_region_calculation_times = json.load(f)
            hybrid_training_times = [float(time) for time in hybrid_prediction_region_calculation_times]
            hybrid_calibration_times[N] = hybrid_training_times
        with open(f"experiment_results/{N}-agent/direct_test_times.json", "r") as f:
            #Todo: double check computation.
            direct_N_test_times = json.load(f)
            direct_N_test_times = [float(time) for time in direct_N_test_times]
            direct_test_times[N] = direct_N_test_times
        with open(f"experiment_results/{N}-agent/indirect_test_times.json", "r") as f:
            indirect_N_test_times = json.load(f)
            indirect_N_test_times = [float(time) for time in indirect_N_test_times]
            indirect_test_times[N] = indirect_N_test_times
        with open(f"experiment_results/{N}-agent/hybrid_test_times.json", "r") as f:
            hybrid_N_test_times = json.load(f)
            hybrid_N_test_times = [float(time) for time in hybrid_N_test_times]
            hybrid_test_times[N] = hybrid_N_test_times
    # Plot the calibration time together for the direct method.
    for N in [5, 7, 10]:
        plt.hist(direct_calibration_times[N], bins=20, color = colormap[N])
        average_direct_calibration_time = np.round(np.average(direct_calibration_times[N]), 3)
        plt.axvline(average_direct_calibration_time, label=f"Average Time (N = {N}): {average_direct_calibration_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/direct_prediction_region_calculation_times" + params.plotting_saving_format)
    plt.show()
    # Plot the calibration time together for the indirect method (variant I);.
    for N in [5, 7, 10]:
        plt.hist(indirect_calibration_times[N], bins=20, color = colormap[N])
        average_indirect_calibration_time = np.round(np.average(indirect_calibration_times[N]), 3)
        plt.axvline(average_indirect_calibration_time, label=f"Average Time (N = {N}): {average_indirect_calibration_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/indirect_prediction_region_calculation_times" + params.plotting_saving_format)
    plt.show()
    # Plot the calibration time together for the indirect method (variant II):
    for N in [5, 7, 10]:
        plt.hist(hybrid_calibration_times[N], bins=20, color = colormap[N])
        average_hybrid_calibration_time = np.round(np.average(hybrid_calibration_times[N]), 3)
        plt.axvline(average_hybrid_calibration_time, label=f"Average Time (N = {N}): {average_hybrid_calibration_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/hybrid_prediction_region_calculation_times" + params.plotting_saving_format)
    plt.show()
    # Plot the test time together for the direct method:
    for N in [5, 7, 10]:
        plt.hist(direct_test_times[N], bins=20, color = colormap[N])
        average_direct_test_time = np.round(np.average(direct_test_times[N]), 3)
        plt.axvline(average_direct_test_time, label=f"Average Time (N = {N}): {average_direct_test_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/direct_test_times" + params.plotting_saving_format)
    plt.show()
    # Plot the test time together for the indirect method (variant I):
    for N in [5, 7, 10]:
        plt.hist(indirect_test_times[N], bins=20, color = colormap[N])
        average_indirect_test_time = np.round(np.average(indirect_test_times[N]), 3)
        plt.axvline(average_indirect_test_time, label=f"Average Time (N = {N}): {average_indirect_test_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/indirect_test_times" + params.plotting_saving_format)
    plt.show()
    # Plot the test time together for the indirect method (variant II):
    for N in [5, 7, 10]:
        plt.hist(hybrid_test_times[N], bins=20, color = colormap[N])
        average_hybrid_test_time = np.round(np.average(hybrid_test_times[N]), 3)
        plt.axvline(average_hybrid_test_time, label=f"Average Time (N = {N}): {average_hybrid_test_time}", color = colormap[N])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"plots/all-agents/hybrid_test_times" + params.plotting_saving_format)
    plt.show()


if __name__ == '__main__':
    main()