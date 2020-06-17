import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.externals import joblib
import matplotlib.pyplot as plt 

# Download data
df = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit.py/master/examples/Bio/bio_100Hz.csv")

# Plot it
df.plot()
plt.title('All the data ploted')
plt.savefig('graphs/alldata')

# Process the signals
bio = nk.bio_process(ecg=df["ECG"], rsp=df["RSP"], eda=df["EDA"], add=df["Photosensor"], sampling_rate=100)

# Plot the processed dataframe, normalizing all variables for viewing purpose
nk.z_score(bio["df"]).plot()
plt.title('Signals plotted')
plt.savefig('graphs/signalsplot')

# Get average quality
print("AVG quality:")
print(bio["ECG"]["Average_Signal_Quality"]) 

# Plot all the heart beats
pd.DataFrame(bio["ECG"]["Cardiac_Cycles"]).plot(legend=False)  
plt.title('Heart beats')
plt.savefig('graphs/hbeats')

# A large number of HRV indices can be found
print("HRV indices: ")
print(bio["ECG"]["HRV"])

# RSA (respiratory sinus arrithmia) algorithm -> P2T
nk.z_score(bio["df"][["ECG_Filtered", "RSP_Filtered", "RSA"]])[1000:2500].plot()
plt.title('RSA algorithm')
plt.savefig('graphs/rsaalg')

#define condition list
condition_list = ["Negative", "Neutral", "Neutral", "Negative"]

#dict containing onsets and durations of each event --> should be 1
events = nk.find_events(df["Photosensor"], cut="lower")
print("event finding")
print(events)

#create_epoch --> epochs of data corresponding to each event (since is 1 event, is the epoch[0])
epochs = nk.create_epochs(bio["df"], events["onsets"], duration=700, onset=-100)
nk.z_score(epochs[0][["ECG_Filtered", "EDA_Filtered", "Photosensor"]]).plot()
plt.title('Epoch')
plt.savefig('graphs/epoch')

# itereate through the epochs and store the interesting results in a new dict that will be, at the end, converted to a dataframe
data = {}  # Initialize an empty dict
for epoch_index in epochs:
    data[epoch_index] = {}  # Initialize an empty dict for the current epoch
    epoch = epochs[epoch_index]

    # ECG
    baseline = epoch["ECG_RR_Interval"].loc[-100:0].mean()  # Baseline
    rr_max = epoch["ECG_RR_Interval"].loc[0:400].max()  # Maximum RR interval
    data[epoch_index]["HRV_MaxRR"] = rr_max - baseline  # Corrected for baseline

    # EDA - SCR
    scr_max = epoch["SCR_Peaks"].iloc[0:600].max()  # Maximum SCR peak
    if np.isnan(scr_max):
        scr_max = 0  # If no SCR, consider the magnitude, i.e.  that the value is 0
    data[epoch_index]["SCR_Magnitude"] = scr_max

data = pd.DataFrame.from_dict(data, orient="index")  # Convert to a dataframe
data["Condition"] = condition_list  # Add the conditions
print("data:")
print(data)
