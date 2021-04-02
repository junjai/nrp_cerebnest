# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:31:49 2020

@author: jinoue
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import pandas as pd
import os
from scipy.stats import wilcoxon, kruskal
#%%
    
class ExperimentData():
    def __init__(self, parameters, clock, eye, target, error, dcn_output, dt=0.01, T=1.6, k=20, nTrials=35, max_time=4000):
        self.LTP1 = parameters['LTP1'][0]
        self.LTD1 = parameters['LTD1'][0]
        self.LTP2 = parameters['LTP2'][0]
        self.LTD2 = parameters['LTD2'][0]
        self.LTP3 = parameters['LTP3'][0]
        self.LTD3 = parameters['LTD3'][0]
        self._eye = eye
        self._target = target
        self._clock = clock
        self.nTrials = nTrials
        self.dt = dt
        self.T = T
        self.k = k
        self.duration = 2/k
        self.saccade_duration = int(2 / k // dt)
        self.max_time = max_time
    def findOnset(self):
        if self.nTrials > self._clock['time'].iloc[-1]//(self.T):
            self.nTrials = int(self._clock['time'].iloc[-1]//self.T)
        align = np.where(self._clock['trial']==1)[0]
        align = align[align < self.max_time]
        onset = np.zeros([2, self.nTrials])
        offset = np.zeros([2, self.nTrials])
        for i in range(self.nTrials):
            onset[0, i] = align[i] - int(self.T/self.dt/2)
            onset[1, i] = align[i] 
        offset = onset + self.saccade_duration
        offset[onset + self.saccade_duration >= self._target.shape[0]] = self._target.shape[0]-1

        self.align = align
        self.onset = onset
        self.offset = offset
    def correctTarget(self, target):
        distance = 0.4
        target_corrected = -np.rad2deg(-np.arctan(-target['x']/distance))
        self.target = target_corrected
    def correctEyepos(self, eye):
        eyepos_corrected = -45*eye['eye angle']
        self.eye = eyepos_corrected
    def findError(self):
        target = self.target
        eye = self.eye
        error = np.zeros([2, self.nTrials])
        for i in range(self.nTrials):
            error[0, i] = target[self.offset[0, i]] + eye[self.offset[0, i]]
            error[1, i] = target[self.offset[1, i]] + eye[self.offset[1, i]]
        self.error = error
    def getDCNOutput(self, dcn_output):
        output = dcn_output['output']
        L = int(self.T//self.dt/4)
        output_aligned = np.zeros([2, L+1, self.nTrials])
        print(output_aligned.shape, self.onset.shape)
        output_aligned[0, :, :] = alignSpikes(output, self.onset[0, :].astype(int), 0, L).T
        output_aligned[1, :, :] = alignSpikes(output, self.onset[1, :].astype(int), 0, L).T
        self.output = output_aligned
    def addWeights(self, weight1=None, weight2=None, weight3=None):
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
    def process(self):
        self.findOnset()
        self.correctTarget(self._target)
        self.correctEyepos(self._eye)
        self.findError()
#         self.getDCNOutput(dcn_output)
        
#     def plot(self):
        

        
def RMS(x):
    rms = np.sqrt(np.mean(x**2))
    return rms

def nVar(x, n):
    v = np.var(x[-n:-1])
    return v

def alignSpikes(spikes, align, l1, l2):
    n = len(align)
    spikes_aligned = np.zeros((n, l1+l2+1))
    for i in range(n):
        if align[i]-l1 < 0:
            spike_of_i = np.ones(l1+l2+1) * np.nan
            spike_of_i[l1-align[i]:] = spikes[:align[i]+l2+1]
            spikes_aligned[i, :] = spike_of_i
        elif align[i] + l2 > len(spikes):
            spike_of_i = np.ones(l1+l2+1) * np.nan
            spike_of_i[:-1-(align[i]+l2-len(spikes))] = spikes[align[i]-l1:]
            spikes_aligned[i, :] = spike_of_i
        else:
            print(align[i]-l1,align[i]+l2+1, spikes_aligned.shape, spikes[align[i]-l1:align[i]+l2+1].shape)
            spikes_aligned[i, :] = spikes[align[i]-l1:align[i]+l2+1]
    
    return spikes_aligned

def readWeights(filepath):
    with open(filepath) as f:
        weights = f.read()

    weights = weights.replace('\n', '').replace('Simulation_reset', '\n').replace(')', ')\n')
    weights = pd.DataFrame([x.split(',') for x in weights.split('\n')])
    header = weights.iloc[0]
    weights = weights[1:-2]
    weights.columns = header
    return weights

def getWeights(df):
    weights = df['weight']
    nTrials = df.shape[0]
    weights_mat = np.array([])
    weights_mat = np.array([float(x) for x in weights.iloc[0].replace(')', '').replace('(', '').split(' ')])
    for i in range(1, nTrials):
        weights_mat = np.vstack([weights_mat, np.array([float(x) for x in weights.iloc[i].replace(')', '').replace('(', '').split(' ')])])
    return weights_mat
#%%
    
def loadData(foldername, nTrials=25, nLast=16, absPath=False):
    if not absPath:
        directory = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
        supfolderpath = directory + '\\data\\' + foldername + '\\'
    else:
        supfolderpath = foldername
    print('supfolder: ', supfolderpath)
    experiments_ = os.listdir(supfolderpath)
    experiments = [s for s in experiments_ if 'csv_records' in s]
    
    nExperiments = len(experiments)
    print(experiments, nExperiments)
    allData = pd.DataFrame(columns=['LTP1', 'LTD1', 'LTP2', 'LTD2', 'LTP3', 'LTD3', 'RMS', 'Variance', 'RMS last'])
    experimentData = np.array([])
    
    for i in range(nExperiments):
        fileid = i
        folderpath = supfolderpath  + experiments[fileid]
    #     folderpath = '../evaluation3/' + experiments[fileid]
        print(i, '/', nExperiments, ': recordings:', folderpath)
        all_spikes = pd.read_csv(folderpath + '/all_spikes.csv')
        clock = pd.read_csv(folderpath + '/clock.csv')
        eye_pos = pd.read_csv(folderpath + '/eye_angle.csv')
        eye_vel = pd.read_csv(folderpath + '/eye_velocity.csv')
        eye_state = pd.read_csv(folderpath + '/eye_state.csv')
        target = pd.read_csv(folderpath + '/target_position.csv')
        error = pd.read_csv(folderpath + '/error.csv')
        true_error = pd.read_csv(folderpath + '/true_error.csv')
        dcn_output = pd.read_csv(folderpath + '/DCN_output.csv')
        params = pd.read_csv(folderpath + './brain_parameters.csv')
        dt = params.dt
        min_len = np.min([clock.shape[0], eye_pos.shape[0], target.shape[0], true_error.shape[0]])
        clock = clock.iloc[:min_len]
        eye_pos = eye_pos.iloc[:min_len]
        target = target.iloc[:min_len]
        true_error = true_error.iloc[:min_len]
        
        try:
            weight1 = readWeights(folderpath + './weights1.csv')
            weight_mat1 = getWeights(weight1)
        except:
            weight_mat1 = []
        try:
            weight2 = readWeights(folderpath + './weights2.csv')
            weight_mat2 = getWeights(weight2)
        except:
            weight_mat2 = []
        try:
            weight3 = readWeights(folderpath + './weights3.csv')
            weight_mat3 = getWeights(weight3)
        except:
            weight_mat3 = []
        
        ex = ExperimentData(params, clock, eye_pos, target, error, dcn_output, dt, nTrials=nTrials)
        ex.process()
        ex.addWeights(weight1=weight_mat1, weight2=weight_mat2, weight3=weight_mat3)
        experimentData = np.append(experimentData, ex)
        ltp1 = ex.LTP1
        ltd1= -ex.LTD1
        ltp2 = ex.LTP2
        ltd2= -ex.LTD2
        ltp3 = ex.LTP3
        ltd3= -ex.LTD3
        rms = (RMS(ex.error[0,:]) + RMS(ex.error[1, :]))/2
        vLast = (nVar(ex.error[0, :], nLast) + nVar(ex.error[1, :], nLast))/2
        rms_last = (RMS(ex.error[0,-nLast:-1]) + RMS(ex.error[1, -nLast:-1]))/2
        tmp_se = pd.Series([ltp1, ltd1, ltp2, ltd2, ltp3, ltd3, rms, vLast, rms_last], index=allData.columns )
        allData = allData.append( tmp_se, ignore_index=True )
        
    return allData, experimentData

#%%
def plotError(i, ex, fig, gs, col, allData, param, nAdapt=16):
    ax[i//col, i%col] = fig.add_subplot(gs[i//col, i%col])
    ax[i//col, i%col].plot(ex[i].error[0,:], label='CF')
    ax[i//col, i%col].plot(ex[i].error[1,:], label='CP')
    ax[i//col, i%col].legend
    ax[i//col, i%col].axhline(0, c='k')
    ax[i//col, i%col].axvline(nAdapt, c='k')
    ax[i//col, i%col].set_xlim([0, 40/1.6])
    if len(param) > 0:
        ax[i//col, i%col].set_title(param[0]+'={0}, '.format(allData[param[0]][i])+param[1]+'={0}'.format(allData[param[1]][i]))
    else:
        ax[i//col, i%col].set_title('no plasticity')
        
def plotWeights(i, ex, fig, gs, col, weights, allData, param):

    ax[i//col, i%col] = fig.add_subplot(gs[i//col, i%col])
    m = getattr(ex[i], weights).mean(1)
    std = getattr(ex[i], weights).std(1)
    trials = np.arange(len(m))
    ax[i//col, i%col].fill_between(trials, m-std, m+std, label='CP', alpha=0.4)
    ax[i//col, i%col].plot(trials, m, label='CF')
    ax[i//col, i%col].axvline(16*2, c='k')
    ax[i//col, i%col].legend
    ax[i//col, i%col].set_title(param[0]+'={0}, '.format(allData[param[0]][i])+param[1]+'={0}'.format(allData[param[1]][i]))

def plotMeanError(ex, allData, param, maxTrials=25, plot=True):
    
    nExperiments = len(ex)
    error_all = np.zeros([nExperiments, maxTrials, 2])
    for i in range(nExperiments):
        error_all[i, :, 0] = ex[i].error[0,:maxTrials]
        error_all[i, :, 1] = ex[i].error[1,:maxTrials] 
    if plot:        
        plt.plot(range(maxTrials), error_all[:,:,0].mean(0))
        plt.plot(range(maxTrials), error_all[:,:,1].mean(0))
        plt.fill_between(range(maxTrials), error_all[:,:,0].mean(0)-error_all[:,:,0].std(0), error_all[:,:,0].mean(0)+error_all[:,:,0].std(0), alpha=0.4)
        plt.fill_between(range(maxTrials), error_all[:,:,1].mean(0)-error_all[:,:,1].std(0), error_all[:,:,1].mean(0)+error_all[:,:,1].std(0), alpha=0.4)
        plt.axhline(0, c='k')
        if len(param)>0: 
            plt.title(param[0]+'={}, '.format(allData[param[0]][0])+ param[1]+'={0}, '.format(allData[param[1]][0])+ 'n={0}'.format(nExperiments))
        else:
            plt.title('n={0}'.format(nExperiments))
    
    return error_all
#%% load data
allData1, experimentData1 = loadData('plast1')
#%%
allData2, experimentData2 = loadData('plast1_2')
#%%
allData3, experimentData3 = loadData('plast1_3')
#%%
allData0, experimentData0 = loadData('control') 
#%%
#absPath = 'C:\\Users\\jinoue\\Documents\\Milano\\Analysis\\Saccade_adaptation\\tuning\\'
#allData_tuning, experimentData_tuning = loadData(absPath, nTrials=36, nLast=16, absPath=True)
#%% plot end point error
row = 2
col = 5
fig = plt.figure(figsize=(20, 4*row))
gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
ax = np.empty([row, col], dtype=object)
param0 = ['ctrl', 'ctrl']
for i in range(10):
    plotError(i, experimentData0, fig, gs, col, allData0, [])
 
fig = plt.figure(figsize=(20, 4*row))
gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
ax = np.empty([row, col], dtype=object)
param1 = ['LTP1', 'LTD1']
for i in range(10):
    plotError(i, experimentData1, fig, gs, col, allData1, param1)
    
fig = plt.figure(figsize=(20, 4*row))
gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
ax = np.empty([row, col], dtype=object)
param2 = ['LTP2', 'LTD2']
for i in range(10):
    plotError(i, experimentData2, fig, gs, col, allData2, param2)
    
fig = plt.figure(figsize=(20, 4*row))
gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
ax = np.empty([row, col], dtype=object)
param3 = ['LTP3', 'LTD3']
for i in range(10):
    plotError(i, experimentData3, fig, gs, col, allData3, param3)

#%% plot weights
#row = 2
#col = 5
#fig = plt.figure(figsize=(20, 4*row))
#gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
#ax = np.empty([row, col], dtype=object)
##param1 = ['LTP1', 'LTD1']
##for i in range(10):
##    plotWeights(i, experimentData1, fig, gs, col, 'weight1', allData1, param1)
#
#fig = plt.figure(figsize=(20, 4*row))
#gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
#ax = np.empty([row, col], dtype=object)
##param2 = ['LTP2', 'LTD2']
##for i in range(10):
##    plotWeights(i, experimentData2, fig, gs, col, 'weight2', allData2, param2)
#
#fig = plt.figure(figsize=(20, 4*row))
#gs = fig.add_gridspec(row, col, height_ratios=np.ones(row))
#ax = np.empty([row, col], dtype=object)
##param2 = ['LTP2', 'LTD2']
##for i in range(10):
##    plotWeights(i, experimentData3, fig, gs, col, 'weight3', allData3, param3)

#%% plot end point error plast1 & control
maxTrials = 25
N = 10
alpha=0.4
plot = False
if plot:
    plt.figure()
error0 = plotMeanError(experimentData0, allData0, param0, maxTrials=maxTrials, plot=plot)
error1 = plotMeanError(experimentData1, allData1, param1, maxTrials=maxTrials, plot=plot)
error2 = plotMeanError(experimentData2, allData2, param2, maxTrials=maxTrials, plot=plot)
error3 = plotMeanError(experimentData3, allData3, param3, maxTrials=maxTrials, plot=plot)
trials = range(1, maxTrials+1)

m_CF1 = error1[:, :, 0].mean(0)
m_CP1 = error1[:, :, 1].mean(0)
std_CF1 = error1[:, :, 0].std(0)
std_CP1 = error1[:, :, 1].std(0)
m_CF2 = error2[:, :, 0].mean(0)
m_CP2 = error2[:, :, 1].mean(0)
std_CF2 = error2[:, :, 0].std(0)
std_CP2 = error2[:, :, 1].std(0)
m_CF3 = error3[:, :, 0].mean(0)
m_CP3 = error3[:, :, 1].mean(0)
std_CF3 = error3[:, :, 0].std(0)
std_CP3 = error3[:, :, 1].std(0)
m_CF0 = error0[:, :, 0].mean(0)
m_CP0 = error0[:, :, 1].mean(0)
std_CF0 = error0[:, :, 0].std(0)
std_CP0 = error0[:, :, 1].std(0)

if not plot:
    plt.figure()
    plt.plot(trials, m_CF1, label='CF plast1')
    plt.plot(trials, m_CP1, label='CP plast1')
    plt.fill_between(trials, m_CF1-std_CF1, m_CF1+std_CF1, alpha=alpha)
    plt.fill_between(trials, m_CP1-std_CP1, m_CP1+std_CP1, alpha=alpha)
    plt.plot(trials, m_CF0, label='CF control')
    plt.plot(trials, m_CP0, label='CP control')
    plt.fill_between(trials, m_CF0-std_CF0, m_CF0+std_CF0, alpha=alpha)
    plt.fill_between(trials, m_CP0-std_CP0, m_CP0+std_CP0, alpha=alpha)
    plt.axhline(0, c='k')
    plt.xlabel('trials')
    plt.ylabel('error (deg)')
    plt.legend()
    
#%%
nTrials = 5
error_early_CF = np.vstack([error1[:, 0:nTrials, 0].flatten(), error2[:, 0:nTrials, 0].flatten(), error3[:, 0:nTrials, 0].flatten()])
error_early_CP = np.vstack([error1[:, 0:nTrials, 1].flatten(), error2[:, 0:nTrials, 1].flatten(), error3[:, 0:nTrials, 1].flatten()])
error_early_all = np.abs(np.vstack([error1[:, 0:nTrials, :].flatten(), error2[:, 0:nTrials, :].flatten(), error3[:, 0:nTrials, :].flatten()]))

error_late_CF = np.vstack([error1[:, -nTrials:, 0].flatten(), error2[:, -nTrials:, 0].flatten(), error3[:, -nTrials:, 0].flatten()])
error_late_CP = np.vstack([error1[:, -nTrials:, 1].flatten(), error2[:, -nTrials:, 1].flatten(), error3[:, -nTrials:, 1].flatten()])
error_late_all = np.abs(np.vstack([error1[:, -nTrials:, :].flatten(), error2[:, -nTrials:, :].flatten(), error3[:, -nTrials:, :].flatten()]))

names1 = ['early: PF-PC', 'early: PF-PC, MF-DCN', 'early: PF-PC, PC-DCN', ]
df_error_early_CF = pd.DataFrame(error_early_CF.T, columns=names1)
df_error_early_CP = pd.DataFrame(error_early_CP.T, columns=names1)
df_error_early_all = pd.DataFrame(error_early_all.T, columns=names1)
names2 = ['late: PF-PC', 'late: PF-PC, MF-DCN', 'late: PF-PC, PC-DCN']
df_error_late_CF = pd.DataFrame(error_late_CF.T, columns=names2)
df_error_late_CP = pd.DataFrame(error_late_CP.T, columns=names2)
df_error_late_all = pd.DataFrame(error_late_all.T, columns=names2)

#%%
names_all = ['error','phase', 'direction', 'plasticity']
s_CF = pd.Series('CF').repeat(N*nTrials)
s_CP = pd.Series('CP').repeat(N*nTrials)
s_plast0 = pd.Series('control').repeat(N*nTrials)
s_plast1 = pd.Series('plast1').repeat(N*nTrials)
s_plast2 = pd.Series('plast1&2').repeat(N*nTrials)
s_plast3 = pd.Series('plast1&3').repeat(N*nTrials)
s_early = pd.Series('early').repeat(N*nTrials)
s_late = pd.Series('late').repeat(N*nTrials)
#df_all = pd.DataFrame(pd.concat([pd.Series(error1[:, 0:nTrials, 0].flatten()), s_early, s_CF, s_plast1], axis=1), columns=names_all, index=s_CF.index)
df_all = pd.DataFrame({'error': error1[:, 0:nTrials, 0].flatten(), 'phase': s_early, 'direction': s_CF, 'plasticity': s_plast1})
df_all = pd.concat([df_all, pd.DataFrame({'error': error1[:, 0:nTrials, 1].flatten(), 'phase': s_early, 'direction': s_CP, 'plasticity': s_plast1})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error1[:, -nTrials:, 0].flatten(), 'phase': s_late, 'direction': s_CF, 'plasticity': s_plast1})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error1[:, -nTrials:, 1].flatten(), 'phase': s_late, 'direction': s_CP, 'plasticity': s_plast1})])

df_all = pd.concat([df_all, pd.DataFrame({'error': error2[:, 0:nTrials, 0].flatten(), 'phase': s_early, 'direction': s_CF, 'plasticity': s_plast2})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error2[:, 0:nTrials, 1].flatten(), 'phase': s_early, 'direction': s_CP, 'plasticity': s_plast2})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error2[:, -nTrials:, 0].flatten(), 'phase': s_late, 'direction': s_CF, 'plasticity': s_plast2})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error2[:, -nTrials:, 1].flatten(), 'phase': s_late, 'direction': s_CP, 'plasticity': s_plast2})])

df_all = pd.concat([df_all, pd.DataFrame({'error': error3[:, 0:nTrials, 0].flatten(), 'phase': s_early, 'direction': s_CF, 'plasticity': s_plast3})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error3[:, 0:nTrials, 1].flatten(), 'phase': s_early, 'direction': s_CP, 'plasticity': s_plast3})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error3[:, -nTrials:, 0].flatten(), 'phase': s_late, 'direction': s_CF, 'plasticity': s_plast3})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error3[:, -nTrials:, 1].flatten(), 'phase': s_late, 'direction': s_CP, 'plasticity': s_plast3})])

df_all = pd.concat([df_all, pd.DataFrame({'error': error0[:, 0:nTrials, 0].flatten(), 'phase': s_early, 'direction': s_CF, 'plasticity': s_plast0})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error0[:, 0:nTrials, 1].flatten(), 'phase': s_early, 'direction': s_CP, 'plasticity': s_plast0})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error0[:, -nTrials:, 0].flatten(), 'phase': s_late, 'direction': s_CF, 'plasticity': s_plast0})])
df_all = pd.concat([df_all, pd.DataFrame({'error': error0[:, -nTrials:, 1].flatten(), 'phase': s_late, 'direction': s_CP, 'plasticity': s_plast0})])

df_all['abs_error'] = df_all['error'].abs()
#%% boxplot difference of plasticity plast1, 2, 3

color1 = 'w'
color2 = '.25'
row, col = 1, 3
plt.figure(figsize=(16, 8))
plt.subplot(row, col, 1)
sns.boxplot(data=df_error_early_CF, color=color1)
sns.swarmplot(data=df_error_early_CF, c=color2)
plt.title('CF: first {} trials'.format(nTrials))
plt.ylabel('error (deg)')

plt.subplot(row, col, 2)
sns.boxplot(data=df_error_early_CP, color=color1)
sns.swarmplot(data=df_error_early_CP, c=color2)
plt.title('CP: first {} trials'.format(nTrials))
plt.ylabel('error (deg)')

plt.subplot(row, col, 3)
sns.boxplot(data=df_error_early_all, color=color1)
sns.swarmplot(data=df_error_early_all, c=color2)
plt.title('all: first {} trials'.format(nTrials))
plt.ylabel('absolute error (deg)')

#plt.subplot(row, col, 4)
#sns.boxplot(data=df_error_late_CF, color=color1)
#sns.swarmplot(data=df_error_late_CF, c=color2)
#plt.title('CF: last {} trials'.format(nTrials))
#
#plt.subplot(row, col, 5)
#sns.boxplot(data=df_error_late_CP, color=color1)
#sns.swarmplot(data=df_error_late_CP, c=color2)
#plt.title('CP: last {} trials'.format(nTrials))
#
#plt.subplot(row, col, 6)
#sns.boxplot(data=df_error_late_all, color=color1)
#sns.swarmplot(data=df_error_late_all, c=color2)
#plt.title('all: last {} trials'.format(nTrials))

plt.suptitle('difference of plasticity 2 and 3')
#%% compare early and late
df_error_phase_CF = pd.concat([df_error_early_CF[names1[0]], df_error_late_CF[names2[0]]], axis=1)
df_error_phase_CP = pd.concat([df_error_early_CP[names1[0]], df_error_late_CP[names2[0]]], axis=1)
df_error_phase_all = pd.concat([df_error_early_all[names1[0]], df_error_late_all[names2[0]]], axis=1)

row, col = 1, 3
plt.figure(figsize=(16, 8))
plt.subplot(row, col, 1)
sns.boxplot(data=df_error_phase_CF, color=color1)
sns.swarmplot(data=df_error_phase_CF, c=color2)
plt.ylabel('error (deg)')
plt.title('CF')

plt.subplot(row, col, 2)
sns.boxplot(data=df_error_phase_CP, color=color1)
sns.swarmplot(data=df_error_phase_CP, c=color2)
plt.ylabel('error (deg)')
plt.title('CP')

plt.subplot(row, col, 3)
sns.boxplot(data=df_error_phase_all, color=color1)
sns.swarmplot(data=df_error_phase_all, c=color2)
plt.ylabel('absolute error (deg)')
plt.title('all')

plt.suptitle('difference between early and late trials')

#%%
#directory = os.getcwd()
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
supfolderpath = directory + '\\data\\' + 'plast1' + '\\'
print('supfolder: ', supfolderpath)
experiments_ = os.listdir(supfolderpath)
experiments = [s for s in experiments_ if 'csv_records' in s]
fileid = 0
folderpath = supfolderpath  + experiments[fileid]
          
clock = pd.read_csv(folderpath + '/clock.csv')
eye_pos = pd.read_csv(folderpath + '/eye_angle.csv')
eye_vel = pd.read_csv(folderpath + '/eye_velocity.csv')
eye_state = pd.read_csv(folderpath + '/eye_state.csv')
target = pd.read_csv(folderpath + '/target_position.csv')
true_error = pd.read_csv(folderpath + '/true_error.csv')
dcn_output = pd.read_csv(folderpath + '/DCN_output.csv')

t_max = max(clock['time'])
dt = 0.01  
t = np.arange(0, t_max, dt)
T = 1.6
dur = 0.2 

eyepos_corrected = -45*eye_pos['eye angle'][:int(t_max/dt)]  
target_corrected = np.rad2deg(-np.arctan(-target['x'][:int(t_max/dt)]  /0.4))
error_corrected = -true_error['error'][:int(t_max/dt)] 

eyevel = -45 * eye_vel['eye velocity'][:int(t_max/dt)]
dcn = -45 * dcn_output['output'][:int(t_max/dt)] * 0.07
#%% plot behavior 
  
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, eyepos_corrected, label='eye position')
plt.plot(t, target_corrected, label='target position')
plt.plot(t, error_corrected, label='error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, eyevel, label='velocity')
plt.plot(t, dcn, label='DCN correction')
plt.legend()

#%% example of eye trajectory and velocity
iTrial1 = 0
iTrial2 = 17
offset1 = 0.2
offset2 = -0.2
row, col = 2, 2

plt.figure(figsize=(10,10))
plt.subplot(row, col, 1)
plt.axhline(0, c='k')
plt.plot(t, eyepos_corrected, label='eye position')
plt.plot(t, target_corrected, label='target position')
plt.plot(t, error_corrected, label='error')
plt.legend()
plt.xlim([T/2+T*iTrial1-offset1, T*(iTrial1+1)++offset2])
plt.ylabel('deg')
plt.title('{}th trial'.format(iTrial1+1))

plt.subplot(row, col, col+1)
plt.axhline(0, c='k')
plt.plot(t, eyevel, label='velocity')
plt.plot(t, dcn, label='target position')
plt.legend()
plt.xlim([T/2+T*iTrial1-offset1, T*(iTrial1+1)++offset2])
plt.ylabel('deg/s')

plt.subplot(row, col, 2)
plt.axhline(0, c='k')
plt.plot(t, eyepos_corrected, label='eye position')
plt.plot(t, target_corrected, label='target position')
plt.plot(t, error_corrected, label='error')
#plt.legend()
plt.xlim([T/2+T*iTrial2-offset1, T*(iTrial2+1)++offset2])
plt.title('{}th trial'.format(iTrial2+1))

plt.subplot(row, col, col+2)
plt.axhline(0, c='k')
plt.plot(t, eyevel, label='velocity')
plt.plot(t, dcn, label='target position')
#plt.legend()
plt.xlim([T/2+T*iTrial2-offset1, T*(iTrial2+1)++offset2])


#%% 
# statistic early vs late
error_early = np.zeros([2, nTrials*N])
error_late = np.zeros([2, nTrials*N])
error_early[0, :] = abs(error1[:, :nTrials, 0].flatten())
error_early[1, :] = abs(error1[:, :nTrials, 1].flatten())
error_late[0, :] = abs(error1[:, -nTrials:, 0].flatten())
error_late[1, :] = abs(error1[:, -nTrials:, 1].flatten())

p1=wilcoxon(df_all[(df_all['plasticity']=='plast1') & (df_all['phase']=='early')].abs_error,
            df_all[(df_all['plasticity']=='plast1') & (df_all['phase']=='late')].abs_error,
            alternative='greater')
p0=wilcoxon(df_all[(df_all['plasticity']=='control') & (df_all['phase']=='early')].abs_error,
            df_all[(df_all['plasticity']=='control') & (df_all['phase']=='late')].abs_error,
            alternative='greater')

color3 = 'tab:blue'
color4 = 'tab:orange'
ls1 = '-'
ls2 = '--'
alpha1 = 0.4
alpha2 = 0.2
yl = [-0.2, 3.5]
row, col = 2, 2
plt.figure(figsize=(10,10))

plt.subplot(row, col, 2)
plt.plot(trials, m_CF1, label='CF(plast1)', color=color3, linestyle=ls1)
plt.plot(trials, m_CP1, label='CP(plat1)', color=color4, linestyle=ls1)
plt.fill_between(trials, m_CF1-std_CF1, m_CF1+std_CF1, alpha=alpha1, color=color3)
plt.fill_between(trials, m_CP1-std_CP1, m_CP1+std_CP1, alpha=alpha2, color=color4)
plt.plot(trials, m_CF0, label='CF(control)', color=color3, linestyle=ls2)
plt.plot(trials, m_CP0, label='CP(control)', color=color4, linestyle=ls2)
plt.fill_between(trials, m_CF0-std_CF0, m_CF0+std_CF0, alpha=alpha1, color=color3)
plt.fill_between(trials, m_CP0-std_CP0, m_CP0+std_CP0, alpha=alpha2, color=color4)
plt.axhline(0, c='k')
plt.xlabel('trials')
plt.ylabel('error (deg)')
plt.legend()

plt.subplot(row, col, col+1)
sns.boxplot(x='phase', y='abs_error', data=df_all[df_all['plasticity']==('plast1')], color=color1)
sns.swarmplot(x='phase', y='abs_error', data=df_all[df_all['plasticity']==('plast1')], hue='direction', c=color2)
plt.ylabel('absolute error (deg)')
plt.ylim(yl)
plt.title('PF-PC plasticity')

plt.subplot(row, col, col+2)
sns.boxplot(x='phase', y='abs_error', data=df_all[df_all['plasticity']==('control')], color=color1)
sns.swarmplot(x='phase', y='abs_error', data=df_all[df_all['plasticity']==('control')], hue='direction', c=color2)
plt.ylabel('absolute error (deg)')
plt.ylim(yl)
plt.title('control')

df_all[(df_all['plasticity']==('plast1')) & (df_all['phase']=='early')].abs_error.mean()
df_all[(df_all['plasticity']==('plast1')) & (df_all['phase']=='early')].abs_error.std()#/np.sqrt(50)
df_all[(df_all['plasticity']==('plast1')) & (df_all['phase']=='late')].abs_error.mean()
df_all[(df_all['plasticity']==('plast1')) & (df_all['phase']=='late')].abs_error.std()#/np.sqrt(50)
df_all[(df_all['plasticity']==('control')) & (df_all['phase']=='early')].abs_error.mean()
df_all[(df_all['plasticity']==('control')) & (df_all['phase']=='early')].abs_error.std()#/np.sqrt(50)
df_all[(df_all['plasticity']==('control')) & (df_all['phase']=='late')].abs_error.mean()
df_all[(df_all['plasticity']==('control')) & (df_all['phase']=='late')].abs_error.std()#/np.sqrt(50)
#%%
#p2_early = kruskal(df_all[(df_all['plasticity']=='plast1') & (df_all['direction']=='CF') & (df_all['phase']=='early')].error.values,
#                     df_all[(df_all['plasticity']=='plast1&2') & (df_all['direction']=='CF') & (df_all['phase']=='early')].error.values,
#                     df_all[(df_all['plasticity']=='plast1&3') & (df_all['direction']=='CF') & (df_all['phase']=='early')].error.values)
#
#p2_late = kruskal(df_all[(df_all['plasticity']=='plast1') & (df_all['direction']=='CF') & (df_all['phase']=='late')].error.values,
#                     df_all[(df_all['plasticity']=='plast1&2') & (df_all['direction']=='CF') & (df_all['phase']=='late')].error.values,
#                     df_all[(df_all['plasticity']=='plast1&3') & (df_all['direction']=='CF') & (df_all['phase']=='late')].error.values)

p2_early = kruskal(df_all[(df_all['plasticity']=='plast1') & (df_all['phase']=='early')].error.values,
                     df_all[(df_all['plasticity']=='plast1&2') & (df_all['phase']=='early')].error.values,
                     df_all[(df_all['plasticity']=='plast1&3') & (df_all['phase']=='early')].error.values)

p2_late = kruskal(df_all[(df_all['plasticity']=='plast1') & (df_all['phase']=='late')].error.values,
                     df_all[(df_all['plasticity']=='plast1&2') & (df_all['phase']=='late')].error.values,
                     df_all[(df_all['plasticity']=='plast1&3') & (df_all['direction']=='CF') & (df_all['phase']=='late')].error.values)

#%% comparison of different plasiticities
color5 = 'tab:green'
color6 = 'tab:red'
color7 = 'tab:purple'

row, col = 1, 3

plt.figure(figsize=(16,5))

plt.subplot(row, col, 1)
plt.plot(trials, m_CF1, label='plast1', color=color3, linestyle=ls1)
plt.plot(trials, m_CF2, label='plast1&2', color=color5, linestyle=ls1)
plt.plot(trials, m_CF3, label='plast1&3', color=color6, linestyle=ls1)
plt.fill_between(trials, m_CF1-std_CF1, m_CF1+std_CF1, alpha=alpha2, color=color3)
plt.fill_between(trials, m_CF2-std_CF2, m_CF2+std_CF2, alpha=alpha2, color=color5)
plt.fill_between(trials, m_CF3-std_CF3, m_CF3+std_CF3, alpha=alpha2, color=color6)
plt.axhline(0, c='k')
plt.ylabel('error (deg)')
plt.legend(frameon=False)

plt.subplot(row, col, 2)
sns.boxplot(x='plasticity', y='abs_error', data=df_all[(df_all['phase']=='early') & (df_all['plasticity'] != 'control') & (df_all['direction'] == 'CF')], color=color1)
sns.swarmplot(x='plasticity', y='abs_error', data=df_all[(df_all['phase']=='early') & (df_all['plasticity'] != 'control') & (df_all['direction'] == 'CF')], palette=[color3, color5, color6])
plt.axhline(0, linestyle=':', c='k')
plt.ylabel('absolute error (deg)')
#plt.ylim(yl)
plt.xlabel('')
plt.title('early trials')

plt.subplot(row, col, 3)
sns.boxplot(x='plasticity', y='abs_error', data=df_all[(df_all['phase']=='late') & (df_all['plasticity'] != 'control') & (df_all['direction'] == 'CF')], color=color1)
sns.swarmplot(x='plasticity', y='abs_error', data=df_all[(df_all['phase']=='late') & (df_all['plasticity'] != 'control') & (df_all['direction'] == 'CF')], c=color2, palette=[color3, color5, color6])
plt.axhline(0, linestyle=':', c='k')
plt.ylabel('absolute error (deg)')
#plt.ylim(yl)
plt.xlabel('')
plt.title('late trials')

df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)
df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1&2') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1&2') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)
df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1&3') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='early') & (df_all['plasticity'] == 'plast1&3') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)

df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)
df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1&2') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1&2') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)
df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1&3') & (df_all['direction'] == 'CF')].abs_error.mean()
df_all[(df_all['phase']=='late') & (df_all['plasticity'] == 'plast1&3') & (df_all['direction'] == 'CF')].abs_error.std()#/np.sqrt(25)