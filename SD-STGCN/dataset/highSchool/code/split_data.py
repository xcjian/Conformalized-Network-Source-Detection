import pickle
import os

# This code is used to split the data such that it will not exceed the size of git commit.
prop_model = 'SIR'
Rzero = 2.5  # simulation R0
beta = 0.3   # beta
gamma = 0    # simulation gamma
ns = 13600    # num of sequences
nf = 16      # num of frames
gt = "highSchool"  # graph type
T = 30       # simulation time steps

data_path = f"../data/{prop_model}/"
file_name = f"{prop_model}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire.pickle"

with open(data_path + file_name, 'rb') as f:
    data = pickle.load(f)

# split the data into 2 files and store them
split_folder = f"../data/{prop_model}/split/"
if not os.path.exists(split_folder):
    os.makedirs(split_folder)
data1 = (data[0][:len(data[0])//2], data[1][:len(data[1])//2])
data2 = (data[0][len(data[0])//2:], data[1][len(data[1])//2:])
with open(split_folder + file_name.replace('.pickle', '_1.pickle'), 'wb') as f:
    pickle.dump(data1, f)
with open(split_folder + file_name.replace('.pickle', '_2.pickle'), 'wb') as f:
    pickle.dump(data2, f)
print('ok')