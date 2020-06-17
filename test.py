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
bio = nk.bio_process(ecg=df["ECG"], sampling_rate=100)
# Plot the processed dataframe, normalizing all variables for viewing purpose
nk.z_score(bio["df"]).plot()
plt.title('ECG plotted')
plt.savefig('graphs/ecgplot')

print(bio["ECG"]["Average_Signal_Quality"]) # Get average quality

# Plot all the heart beats
pd.DataFrame(bio["ECG"]["Cardiac_Cycles"]).plot(legend=False)  
plt.title('Heart beats')
plt.savefig('graphs/hbeats')

# A large number of HRV indices can be found
print(bio["ECG"]["HRV"])

