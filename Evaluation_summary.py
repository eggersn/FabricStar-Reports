import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


# Read csv
asset_sizes = ["100", "1000", "2000", "4000", "8000", "16000"]

hyperledger_path = ["TransactionLoad/Small_Network/Create_Assets_{}/Hyperledger_Evaluation.csv".format(asset_size) for asset_size in asset_sizes]
fabricstar_path = ["TransactionLoad/Small_Network/Create_Assets_{}/FabricStar_Evaluation.csv".format(asset_size) for asset_size in asset_sizes]

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

def getBestIndex(latencies, throughput):
    max_index=0
    max_throughput=0
    for i in range(len(latencies)):
        if throughput[i] > max_throughput and latencies[i] <= 2:
            max_index = i
            max_throughput = throughput[i]

    return max_index

def getElement(list, index):
    return list[index]
        
y_values_hyperledger_throughput = []
y_values_hyperledger_error1 = []
y_values_hyperledger_error2 = []
y_values_hyperledger_min_latency = []
y_values_hyperledger_avg_latency = []
y_values_hyperledger_max_latency = []
y_values_fabricstar_throughput = []
y_values_fabricstar_error1 = []
y_values_fabricstar_error2 = []
y_values_fabricstar_min_latency = []
y_values_fabricstar_avg_latency = []
y_values_fabricstar_max_latency = []
hyperledger_indizes = []
fabricstar_indizes = []

for i in range(len(hyperledger_path)):    
    benchmark_names, hyperledger_values, hyperledger_avg, hyperledger_sd = getData(hyperledger_path[i])
    _, fabricstar_values, fabricstar_avg, fabricstar_sd = getData(fabricstar_path[i])

    hyperledger_index = getBestIndex([hyperledger_avg[benchmark]["Avg Latency (s)"] for benchmark in hyperledger_values], [hyperledger_avg[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values])
    fabricstar_index = getBestIndex([fabricstar_avg[benchmark]["Avg Latency (s)"] for benchmark in fabricstar_values], [fabricstar_avg[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values])

    hyperledger_indizes.append(hyperledger_index)
    fabricstar_indizes.append(fabricstar_index)

    y_values_hyperledger_throughput.append(getElement(list([hyperledger_avg[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values]), hyperledger_index))
    y_values_hyperledger_error1.append(getElement([hyperledger_avg[benchmark]["Throughput (TPS)"] + hyperledger_sd[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values], hyperledger_index))
    y_values_hyperledger_error2.append(getElement([hyperledger_avg[benchmark]["Throughput (TPS)"] - hyperledger_sd[benchmark]["Throughput (TPS)"] for benchmark in hyperledger_values], hyperledger_index))   

    y_values_hyperledger_min_latency.append(getElement([hyperledger_avg[benchmark]["Min Latency (s)"] for benchmark in hyperledger_values], hyperledger_index))
    y_values_hyperledger_avg_latency.append(getElement([hyperledger_avg[benchmark]["Avg Latency (s)"] for benchmark in hyperledger_values], hyperledger_index))
    y_values_hyperledger_max_latency.append(getElement([hyperledger_avg[benchmark]["Max Latency (s)"] for benchmark in hyperledger_values], hyperledger_index))

    y_values_fabricstar_throughput.append(getElement([fabricstar_avg[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values], fabricstar_index))
    y_values_fabricstar_error1.append(getElement([fabricstar_avg[benchmark]["Throughput (TPS)"] + fabricstar_sd[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values], fabricstar_index))
    y_values_fabricstar_error2.append(getElement([fabricstar_avg[benchmark]["Throughput (TPS)"] - fabricstar_sd[benchmark]["Throughput (TPS)"] for benchmark in fabricstar_values], fabricstar_index))   

    y_values_fabricstar_min_latency.append(getElement([fabricstar_avg[benchmark]["Min Latency (s)"] for benchmark in fabricstar_values], fabricstar_index))
    y_values_fabricstar_avg_latency.append(getElement([fabricstar_avg[benchmark]["Avg Latency (s)"] for benchmark in fabricstar_values], fabricstar_index))
    y_values_fabricstar_max_latency.append(getElement([fabricstar_avg[benchmark]["Max Latency (s)"] for benchmark in fabricstar_values], fabricstar_index))

# Input
x_values = [100, 1000, 2000, 4000, 8000, 16000]
bar_width = 350

# Plot Settings
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xticks(x_values)
ax1.grid(zorder=0, color="grey")
ax1.set_xlabel("Transaction Load", fontsize=18)
ax1.set_ylabel("Throughput [TPS]", fontsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

ax2.set_ylabel("Latency (Min/Avg/Max) [sec]", fontsize=18)
ax2.set_ylim([0, 15])

# hyperledger value plots
# Throughput
ax1.plot(x_values, y_values_hyperledger_throughput, color="red", zorder=5, linewidth=3)
ax1.errorbar(x_values, np.array(y_values_hyperledger_throughput), np.array(y_values_hyperledger_error1)-np.array(y_values_hyperledger_throughput), fmt='.k', color="black", elinewidth=3, capsize=5)
hyperledger_label,  = ax1.plot(x_values, y_values_hyperledger_throughput, 'D', color="red", zorder=5)
ax1.fill_between(x_values, y_values_hyperledger_error1, y_values_hyperledger_error2, color="lightcoral", zorder=5, alpha=0.5)
#latency
ax2.bar(np.array(x_values)-bar_width/2, y_values_hyperledger_max_latency, width=bar_width, color="lightsteelblue", align="center", alpha=0.8, zorder=1)
hyperledger_label_bar = ax2.bar(np.array(x_values)-bar_width/2, y_values_hyperledger_avg_latency, width=bar_width, color="blue", align="center", alpha=0.8, zorder=1)
ax2.bar(np.array(x_values)-bar_width/2, y_values_hyperledger_min_latency, width=bar_width, color="darkblue", align="center", alpha=0.8, zorder=1)
# fabricstar value plots
# throughput
ax1.plot(x_values, y_values_fabricstar_throughput, color="green", zorder=10, linewidth=3)
ax1.errorbar(x_values, np.array(y_values_fabricstar_throughput), np.array(y_values_fabricstar_error1)-np.array(y_values_fabricstar_throughput), fmt='.k', color="black", elinewidth=3, capsize=5)
fabricstar_label, = ax1.plot(x_values, y_values_fabricstar_throughput, 'D', color="green", zorder=10)
ax1.fill_between(x_values, y_values_fabricstar_error1, y_values_fabricstar_error2, color="lightgreen", zorder=10, alpha=0.5)
#latency
ax2.bar(np.array(x_values)+bar_width/2, y_values_fabricstar_max_latency, width=bar_width, color="navajowhite", align="center", alpha=0.8, zorder=3)
fabricstar_label_bar = ax2.bar(np.array(x_values)+bar_width/2, y_values_fabricstar_avg_latency, width=bar_width, color="goldenrod", align="center", alpha=0.8, zorder=3)
ax2.bar(np.array(x_values)+bar_width/2, y_values_fabricstar_min_latency, width=bar_width, color="darkgoldenrod", align="center", alpha=0.8, zorder=3)

plt.legend([hyperledger_label, hyperledger_label_bar, fabricstar_label, fabricstar_label_bar], ["Hyperledger Throughput","Hyperledger Latency (Min/Avg/Max)", "Fabric* Throughput", "Fabric* Latency (Min/Avg/Max)"], loc="upper center", prop={'size': 18})
plt.show()   