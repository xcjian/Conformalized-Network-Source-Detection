# Remarks

Before you start, make sure you have installed Git on your machine.

Please massively use ChatGPT and DeepSeek. It will help you a lot when you meet problems in the code. They can even interpret very long and hard code for you.

Also feel free to approach me when you meet any problems.

# Download the repo
Open git bash in the directory you want to work. Then, run the command

`git clone https://github.com/xcjian/Conformalized-Network-Source-Detection.git`

This will directly download all project files in your working directory.

# Setup the environment

## create a virtual python environment

I will recommend python3.9 or 3.10.

## install the packages required

Assume you are working on a linux workstation.

`cd <working directory>`

`cd SD-STGCN`

The following command will install packages required by GCN:

`pip install -r requirements.txt`

The following command will install packages for generating the SIR realizations:

`pip install ndlib`

The following commands will install packages for the baseline method:

`cd ../` (go back to the project folder)

`cd DSI`

`pip install .`

# Generate the SIR realizations

The script for generating it is `SD-STGCN/dataset/highSchool/code`.

the file `run_T.sh` provides a compact way to generate SIR realizations.

Example of use:

`./run_T.sh 16 4000 2.5 0.3 0`

This will generate 4000 realizations of SIR with R0 = 2.5, beta = 0.3, gamma = 0. Here gamma = 0 means nobody will recover, so it's actually a SI model. From this file you can see that it tells the computer to execute a python command, running the python script `sim_T.py`. Go into `sim_T.py `, you will find the place where the SIR realizations are generated:

`sim = SIR(N, g, beta, gamma, min_outbreak_frac=f0)`

You can go into this SIR function to see how we can modify it so that multiple sources can be generated.

# Use Git to store all your updates

Below are several git commands that I usually use.

after you have made some changes:

`git add <file name>`
This allows you to store the changes on a file.

`git commit -m 'experiment on multisource problem'`
After the `git add` command, use `git commit` to store all files at this time point. Git will store all your commits.

`git push orgin <branch name>`
After the `git commit` command, use `git push` to upload the commits to github repository.

`git branch <branch name>`
Create a new branch. This allows you to work on a particular feature first while maintaining the main branch ("master") unchanged.

`git checkout <branch name>`
After creating a branch, use `git checkout` to switch your working thread to the new branch. You can now begin to work on the new branch, without affecting the original branch (e.g., "master").

