import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


# Read csv
hyperledger_path = "MaxTPS/MaxTPS100-2k/FabricStar_Evaluation.csv"
fabricstar_path = "MaxTPS/MaxTPS100-2k/Hyperledger_Evaluation.csv"
hyperledger_path1 = "MaxTPS/MaxTPS4k-16k/FabricStar_Evaluation.csv"
fabricstar_path1 = "MaxTPS/MaxTPS4k-16k/Hyperledger_Evaluation.csv"

def getData(path):
    df = pd.read_csv(path, sep=", ")

    benchmark_names = []
    values = {}
    avg = {}
    sd = {}

    names_found = False

    # Reformat csv
    for index, row in df.iterrows():
        if row["Name"] == "Benchmark 1":
            names_found = True
        if not names_found and index != 0:
            benchmark_names.append(row["Name"])
            values[row["Name"]] = {"Succ": [], "Fail": [], "Send Rate (TPS)": [], "Max Latency (s)": [], "Min Latency (s)": [], "Avg Latency (s)": [], "Throughput (TPS)": []}
            avg[row["Name"]] = {"Succ": 0, "Fail": 0, "Send Rate (TPS)": 0, "Max Latency (s)": 0, "Min Latency (s)": 0, "Avg Latency (s)": 0, "Throughput (TPS)": 0}
            sd[row["Name"]] = {"Succ": 0, "Fail": 0, "Send Rate (TPS)": 0, "Max Latency (s)": 0, "Min Latency (s)": 0, "Avg Latency (s)": 0, "Throughput (TPS)": 0}

        if row["Name"].split()[0] == "Benchmark":
            continue 

        for column in df.columns.values:
            if column != "Name":
                values[row["Name"]][column].append(row[column])

    # Compute Average
    for benchmark in values:
        for category in values[benchmark]:
            avg[benchmark][category] = 1/len(values[benchmark][category]) * sum(values[benchmark][category])
            sd[benchmark][category] = math.sqrt(1/len(values[benchmark][category]) * sum((x - avg[benchmark][category])**2 for x in values[benchmark][category]))

    return benchmark_names, values, avg, sd
    
benchmark_names, hyperledger_values, hyperledger_avg, hyperledger_sd = getData(hyperledger_path)
_, fabricstar_values, fabricstar_avg, fabricstar_sd = getData(fabricstar_path)
benchmark_names1, hyperledger_values1, hyperledger_avg1, hyperledger_sd1 = getData(hyperledger_path1)
_, fabricstar_values1, fabricstar_avg1, fabricstar_sd1 = getData(fabricstar_path1)

# Input
x_values = [100, 1000, 2000, 4000, 8000, 16000]
bar_width = 350

# Plot Settings
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xticks(x_values)
ax1.grid(zorder=0, color="grey")
ax1.set_xlabel("Payload Size [byte]")
ax1.set_ylabel("Throughput [TPS]")

ax2.set_ylabel("Latency (Min/Avg/Max) [sec]")
plt.title("Maximum Throughput")
# ax2.set_ylim([0, 15])

# hyperledger value plots
# Throughput
y_values_hyperledger = [hyperledger_avg[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values]
y_error1_hyperledger = [hyperledger_avg[benchmark]["Throughput (TPS)"] + hyperledger_sd[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values]
y_error2_hyperledger = [hyperledger_avg[benchmark]["Throughput (TPS)"] - hyperledger_sd[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values]

y_values_hyperledger1 = [hyperledger_avg1[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values1]
y_error1_hyperledger1 = [hyperledger_avg1[benchmark]["Throughput (TPS)"] + hyperledger_sd1[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values1]
y_error2_hyperledger1 = [hyperledger_avg1[benchmark]["Throughput (TPS)"] - hyperledger_sd1[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values1]

y_values_hyperledger = np.concatenate((y_values_hyperledger, y_values_hyperledger1), axis=None)
y_error1_hyperledger = np.concatenate((y_error1_hyperledger, y_error1_hyperledger1), axis=None)
y_error2_hyperledger = np.concatenate((y_error2_hyperledger, y_error2_hyperledger1), axis=None)

ax1.plot(x_values, y_values_hyperledger, color="red", zorder=5, linewidth=3)
ax1.errorbar(x_values, np.array(y_values_hyperledger), np.array(y_error1_hyperledger)-np.array(y_values_hyperledger), fmt='.k', color="black", elinewidth=3, capsize=5)
hyperledger_label,  = ax1.plot(x_values, y_values_hyperledger, 'D', color="red", zorder=5)
ax1.fill_between(x_values, y_error1_hyperledger, y_error2_hyperledger, color="lightcoral", zorder=5, alpha=0.5)

# Latency
y_values_hyperledger = np.array([hyperledger_avg[benchmark]["Avg Latency (s)"] for benchmark in hyperledger_values])
y_error1_hyperledger = np.array([hyperledger_avg[benchmark]["Max Latency (s)"] for benchmark in hyperledger_values])
y_error2_hyperledger = np.array([hyperledger_avg[benchmark]["Min Latency (s)"] for benchmark in hyperledger_values])

y_values_hyperledger1 = np.array([hyperledger_avg1[benchmark]["Avg Latency (s)"] for benchmark in hyperledger_values1])
y_error1_hyperledger1 = np.array([hyperledger_avg1[benchmark]["Max Latency (s)"] for benchmark in hyperledger_values1])
y_error2_hyperledger1 = np.array([hyperledger_avg1[benchmark]["Min Latency (s)"] for benchmark in hyperledger_values1])

y_values_hyperledger = np.concatenate((y_values_hyperledger, y_values_hyperledger1), axis=None)
y_error1_hyperledger = np.concatenate((y_error1_hyperledger, y_error1_hyperledger1), axis=None)
y_error2_hyperledger = np.concatenate((y_error2_hyperledger, y_error2_hyperledger1), axis=None)

print(y_values_hyperledger)
print(y_error1_hyperledger)
print(y_error2_hyperledger)

ax2.bar(np.array(x_values)-bar_width/2, y_error1_hyperledger, width=bar_width, color="lightsteelblue", align="center", alpha=0.8, zorder=1)
hyperledger_label_bar = ax2.bar(np.array(x_values)-bar_width/2, y_values_hyperledger, width=bar_width, color="blue", align="center", alpha=0.8, zorder=1)
ax2.bar(np.array(x_values)-bar_width/2, y_error2_hyperledger, width=bar_width, color="darkblue", align="center", alpha=0.8, zorder=1)

# fabricstar value plots
y_values_fabricstar = [fabricstar_avg[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values]
y_error1_fabricstar = [fabricstar_avg[benchmark]["Throughput (TPS)"] + fabricstar_sd[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values]
y_error2_fabricstar = [fabricstar_avg[benchmark]["Throughput (TPS)"] - fabricstar_sd[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values]

y_values_fabricstar1 = [fabricstar_avg1[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values1]
y_error1_fabricstar1 = [fabricstar_avg1[benchmark]["Throughput (TPS)"] + fabricstar_sd1[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values1]
y_error2_fabricstar1 = [fabricstar_avg1[benchmark]["Throughput (TPS)"] - fabricstar_sd1[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values1]

y_values_fabricstar = np.concatenate((y_values_fabricstar, y_values_fabricstar1), axis=None)
y_error1_fabricstar = np.concatenate((y_error1_fabricstar, y_error1_fabricstar1), axis=None)
y_error2_fabricstar = np.concatenate((y_error2_fabricstar, y_error2_fabricstar1), axis=None)

ax1.plot(x_values, y_values_fabricstar, color="green", zorder=10, linewidth=3)
ax1.errorbar(x_values, np.array(y_values_fabricstar), np.array(y_error1_fabricstar)-np.array(y_values_fabricstar), fmt='.k', color="black", elinewidth=3, capsize=5)
fabricstar_label, = ax1.plot(x_values, y_values_fabricstar, 'D', color="green", zorder=10)
ax1.fill_between(x_values, y_error1_fabricstar, y_error2_fabricstar, color="lightgreen", zorder=10, alpha=0.5)

# Latency
y_values_fabricstar = np.array([fabricstar_avg[benchmark]["Avg Latency (s)"] for benchmark in fabricstar_values])
y_error1_fabricstar = np.array([fabricstar_avg[benchmark]["Max Latency (s)"] for benchmark in fabricstar_values])
y_error2_fabricstar = np.array([fabricstar_avg[benchmark]["Min Latency (s)"] for benchmark in fabricstar_values])

y_values_fabricstar1 = np.array([fabricstar_avg1[benchmark]["Avg Latency (s)"] for benchmark in fabricstar_values1])
y_error1_fabricstar1 = np.array([fabricstar_avg1[benchmark]["Max Latency (s)"] for benchmark in fabricstar_values1])
y_error2_fabricstar1 = np.array([fabricstar_avg1[benchmark]["Min Latency (s)"] for benchmark in fabricstar_values1])

y_values_fabricstar = np.concatenate((y_values_fabricstar, y_values_fabricstar1), axis=None)
y_error1_fabricstar = np.concatenate((y_error1_fabricstar, y_error1_fabricstar1), axis=None)
y_error2_fabricstar = np.concatenate((y_error2_fabricstar, y_error2_fabricstar1), axis=None)

print(y_values_fabricstar)
print(y_error1_fabricstar)
print(y_error2_fabricstar)

ax2.bar(np.array(x_values)+bar_width/2, y_error1_fabricstar, width=bar_width, color="navajowhite", align="center", alpha=0.8, zorder=3)
fabricstar_label_bar = ax2.bar(np.array(x_values)+bar_width/2, y_values_fabricstar, width=bar_width, color="goldenrod", align="center", alpha=0.8, zorder=3)
ax2.bar(np.array(x_values)+bar_width/2, y_error2_fabricstar, width=bar_width, color="darkgoldenrod", align="center", alpha=0.8, zorder=3)


plt.legend([hyperledger_label, hyperledger_label_bar, fabricstar_label, fabricstar_label_bar], ["Hyperledger Throughput","Hyperledger Latency (Min/Avg/Max)", "Fabric* Throughput", "Fabric* Latency (Min/Avg/Max)"], loc="upper center")
plt.show()    