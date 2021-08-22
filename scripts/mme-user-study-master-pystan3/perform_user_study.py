# Install the following packages (pip3 install <NAME>) before running the study code:
# seaborn
# torch
# pandas
# pystan
# arviz

import pickle
import random
import numpy as np
import pandas as pd
from generate_data import generate_data
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import subprocess
import webbrowser
from machine_education_interactive_test import ts_teach_user_study, rollout_onestep_la, \
    noeducate_rollout_onestep_la

#matplotlib.use('TkAgg') # Using TkAgg to set window position

def cls():
    os.system('cls' if os.name=='nt' else 'clear') # To clear the console

if not os.path.exists('Results'):
    os.makedirs('Results')

#window = plt.get_current_fig_manager().window
#screen_x, screen_y = window.wm_maxsize() # Screen dimensions


#dpi = plt.figure("Test").dpi # DPI used by plt
#plt.close(fig="Test")
#fig_height = round((screen_y - 100)/(2*dpi)) #Target height of each plt figure
#fig_width = fig_height*4/3 # Target width

#Set up the two plt figures on the screen at the start of the study.
#plt.interactive(True) 

fig, (ax1, ax2) = plt.subplots(2, 1)

#
# webbrowser.open("https://forms.gle/AkY5xLrPzxN43Dry7", new=2)
# input("Please complete the consent form. Press ENTER to continue.")
#
# user_id = int(input("Please input a user id"))
# user_group = int(input("Please input the user group (0 or 1 or 2)"))
#
#
# webbrowser.open("https://forms.gle/pvhY9BMs8BToytBw7", new=2)
# input("Please complete the first part of the questionnaire. Press ENTER to continue")
#
#
#
# if user_group == 2:
#     #subprocess.Popen(['open','Material/Full_Tutorial.pdf']) # Open full tutorial for group 2 (pre-trained users)
#     os.system("start Material/Full_Tutorial.pdf")
# else:
#     #subprocess.Popen(['open','Material/Basic_Tutorial.pdf']) # Basic tutorial for group 0 (no-training) and group 1 (AI-trained)
#     os.system("start Material/Basic_Tutorial.pdf")
#
# input("Please go through the tutorial carefully. Press ENTER to continue")
#
#
# input("Please complete the next part of the questionnaire (in your browser window). Press ENTER to continue")

user_id = 123
user_group = 1

#For user group i, the user gets tutoring teacher if groups_tutor_or_not[i] == True.
groups_tutor_or_not = [False, True, True]

#Use the user id as the seed.
np.random.seed(user_id)

### BELOW IS IMPORTANT EXPERIMENT PARAMETERS ###



#Educability and cost. Should be tuned if there are issues.
e = 0.30
theta_1 = 0.5

# Pairs of (nr. of non-collinear variables, nr. of collinear variables).
# In increasing difficulty.
# difficulty = [(8,2), (6,4), (5,5), (4,6), (2,8)]
difficulty = [(5,3), (4,4), (3,5)]

#How many different datasets will the user face? We discussed 5 last time.
n_tasks = len(difficulty)


random.shuffle(difficulty) # Shuffle the order
#FORGET ABOUT THESE
W_typezero=(7.0, 0.0)
W_typeone=(7.0, -7.0)

experiment_results = []
csv_frames = []
for t in range(n_tasks):

    if groups_tutor_or_not[user_group] is True:
        pfunc = rollout_onestep_la
    else:
        pfunc = noeducate_rollout_onestep_la
    cls()
    input("Task #{} out of {}. Press ENTER to begin.".format(t+1,n_tasks))
    #if t == 0:
        # Set up plots before first task.
        #plt.figure("X-Y",figsize=(fig_width, fig_height))
        #plt.get_current_fig_manager().window.wm_geometry("+{}+{}".format(round(screen_x-fig_width*dpi),0)) # move the window
        #plt.figure("X-X",figsize=(fig_width, fig_height))
        #plt.get_current_fig_manager().window.wm_geometry("+{}+{}".format(round(screen_x-fig_width*dpi),round(screen_y/2 + 10))) # move the window

    training_dataset = generate_data(n_noncollinear=difficulty[t][0], n_collinear=difficulty[t][1], n=100)
    test_datasets = [generate_data(n_noncollinear=difficulty[t][0], n_collinear=difficulty[t][1], n=100) for _ in range(10)]
    test_datasets.append(training_dataset)

    pickle_return , csv_return =ts_teach_user_study(ax1=ax1, ax2=ax2, educability=e, dataset=training_dataset,
                                                            W_typezero = W_typezero,
                                                            W_typeone = W_typeone,
                                                            planning_function=pfunc,
                                                            n_interactions=16,
                                                            test_datasets=test_datasets,
                                                            n_training_noncollinear=difficulty[t][0],
                                                            n_training_collinear=difficulty[t][1], theta_1=theta_1,
                                                            theta_2=1 - theta_1, user_id = user_id, group_id = user_group, task_id = t)

    experiment_results.append(pickle_return)
    experiment_results[-1]["order_of_difficulty"] = difficulty[t]
    experiment_results[-1]["tutor_or_not"] = groups_tutor_or_not[user_group]
    experiment_results[-1]["group_id"] = user_group
    experiment_results[-1]["user_id"] = user_id
    experiment_results[-1]["educability"] = e
    experiment_results[-1]["theta_1"] = theta_1

    csv_frames.append(csv_return)
    #plt.figure("X-Y")
    #plt.clf()
    #plt.figure("X-X")
    #plt.clf()
    
    ax1.cla()
    ax2.cla()
    input("End of task. Please take a short break. Press ENTER to continue.")
    # csv_frames.append(difficulty)


with open("Results/participant{}_group{}_{}.csv".format(user_id, user_group,time.time()),
          'wb') as pickle_file:
    pickle.dump(experiment_results, pickle_file)

csv_df = pd.concat(csv_frames, ignore_index = True)
csv_df.to_csv("Results/participant{}_group{}_{}.csv".format(user_id, user_group,time.time()), sep=',')

input("Please complete the final part of the questionnaire (in your browser window). Press ENTER to finish the study.")
