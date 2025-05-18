# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:30:57 2020

@author: Nathan Cross

General functions and other nitty gritties

"""
from os import listdir, mkdir, path
from cfc_func import mean_amp, surface_laplacian, _allnight_ampbin, _allnight_ampbin_per_sub, klentropy
from datetime import datetime, timedelta 
import math 
import matplotlib.pyplot as plt
from numpy import arange
from numpy import (all, argmin, around, array, asarray,  concatenate, cumsum, 
                   empty, format_float_positional, histogram, hstack, isnan, linspace, log, mean, minimum, nan, nanargmax,
                   nanmean, ndarray, ones, pi, reshape, roll, savetxt, sin, squeeze, sqrt, std, sum, transpose, 
                   vstack, where, zeros) # remove asscalar
import pandas as pd
from pandas import concat, DataFrame, ExcelFile, read_csv
from pickle import dump, load
import shutil
import sys
import csv
from tensorpac import Pac
from wonambi import Dataset, graphoelement
from wonambi.attr import Annotations
from wonambi.attr.annotations import create_empty_annotations, Annotations
from wonambi.detect import consensus, DetectSpindle, DetectSlowWave
from wonambi.trans import fetch, get_times
from wonambi.trans.analyze import event_params, export_event_params

""" CODE """


def importit(root_dir, subj, chan, rater, system):
    # 
    # General import script.
    # This script takes input data (edf file, staging file, noise scores)
    # and translates them in the correct format for analysis in Wonambi.
    #
    # Required Inputs:
    # <Subject-ID>.edf = file with EEG recording
    # <Subject-ID>_staging.txt = file with staging information
    # <Subject-ID>_arousals.txt  = file with artefact markings/noise scores
    # noise_scores.xls = file with artefact markings/noise scores (in 5s epochs)
    #
    # Output: 
    # <Subject-ID>.xml = annotations file
    #
    # First the script will check whether the data is in the format of one file
    # per subject, or whether the subject has multiple timepoints (sessions) with 
    # each set of files per timepoint (as per BIDS organisation).
    # 
    """ SET REQUIRED OUTPUT DIRECTORIES """
    check = []
    for i,fname in enumerate(listdir(root_dir + subj)):
        if fname.endswith('.edf') or fname.endswith('.set'):  #check if subject has multiple sessions
            check.append(1)
        else: 
            check.append(0)
    
    if sum(check) > 0: # if subj has no sessions
        if path.exists(root_dir +  subj + r'/wonambi//'):
                print(root_dir +  subj + " already exists")
        else:
                mkdir((root_dir +  subj + r'/wonambi//'))
                mkdir((root_dir +  subj + r'/wonambi/backups//'))
              
        rec_dir = [[root_dir +  subj + r'/']] # records folder
        backup = [[root_dir + subj + r'/wonambi/backups//']] #backup folder
        
    elif sum(check) == 0: # if subj has multiple sessions
        rec_dir = []
        backup = []
        for i,ses in enumerate(listdir(root_dir + subj)):
            if path.exists(root_dir +  subj + '/' + ses + r'/wonambi//'):
                print(root_dir +  subj + '/' + ses + " already exists")
            else:
                mkdir((root_dir +  subj + '/' + ses + r'/wonambi//'))
                mkdir((root_dir +  subj + '/' + ses + r'/wonambi/backups//'))
              
            rec_dir.append([root_dir +  subj + '/' + ses + r'/']) # records folders
            backup.append([root_dir +  subj + '/' + ses + r'/wonambi/backups//']) #backup folders
    
    
    
    """ CREATE NEW ANNOTATIONS FILE AND IMPORT STAGING"""
    for i,rdir in enumerate(rec_dir):
        annot_dir = rdir[0] + r'/wonambi//' # annotations folder
        annot_file = annot_dir + subj + '.xml'
        rec = [file for file in listdir(rdir[0]) if file.endswith(".edf") or file.endswith(".rec") or file.endswith(".set") if not file.startswith(".")][0]
        dataset = Dataset(rdir[0] + rec)
        ch = empty(len(chan), dtype='object')
        for i, cn in enumerate(chan):
            cn=str(cn) # for egi
            try: 
                channum = [dataset.header['chan_name'].index(s) for s in dataset.header['chan_name'] if cn in s]
                chan_ful = dataset.header['chan_name'][channum[0]]
            except Exception as e:
                print(f'WARNING:{cn} not in recording.')
                print(f'Check the electrodes for this recording and try again.')
                continue
            ch[i] =  [s for s in dataset.header['chan_name'] if chan_ful in s][0]
        chan = ndarray.tolist(ch)
        create_empty_annotations(annot_file, dataset)
        annot = Annotations(annot_file)
        rec_start = dataset.header['start_time']
        
        filename = rdir[0] + '/' + subj + '_staging.txt'
        if path.isfile(filename) == False:
            filename = rdir[0] + '/' + subj + '_staging.csv'
        # Read staging file to check for epoch length
        if system == 'alice':
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            try: 
                stage_fmt = float(lines[2].strip())
                if stage_fmt>9:
                    epoch_length = 1
                else:
                    epoch_length = None
            except Exception as e:
                epoch_length = None
        else:
            epoch_length = None
        print(f'Creating annotations for {rec}')
        Annotations.import_staging(annot, filename, source=system, rater_name=rater, 
                                   rec_start=rec_start, staging_start=None, epoch_length=epoch_length,
                                   poor=['Artefact'], as_qual=False)
    
    
    """ IMPORT ARTEFACTS AND SAVE TO ANNOTATIONS FILE """
    for r,rdir in enumerate(rec_dir):
        if system == 'egi':                     #hdEEG has specific artefact rejection format
            noisefile = [file for file in listdir(rdir[0]) if file.endswith("rejepochs.txt") if not file.startswith(".")][0]

            #noisefile = [file for file in listdir(rdir[0]) if file.endswith("rejepochs.txt") if not file.startswith(".")][0]
            if path.isfile(rdir[0] + noisefile):
                noise = pd.read_csv(rdir[0] + noisefile, sep=None, header=None, engine='python')
                artefacts = empty((len(noise),8),dtype=object) #rearrange the data for importing into annotations file
                dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
                i=-1
                for j,val in enumerate(noise[2][1:]):
                        if "reject" in noise[2][j+1]:
                            i+=1
                            artefacts[i,0] = i
                            artefacts[i,1] = noise[0][j+1]
                            artefacts[i,2] = float(artefacts[i,1]) + float(noise[1][j+1])

                            artefacts[i,7] = '(eeg)'        
                arfile = rdir[0]  + r'/wonambi//' + subj + '-artefact_scores.csv'     #save the rearranged events to csv    
                savetxt(arfile, artefacts, fmt="%s", delimiter=',', 
                        header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                a = graphoelement.events_from_csv(arfile) 
                backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                               str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                shutil.copy(annot_file, backup_file) 
                print(f'Importing artefacts from {noisefile}')
                a.to_annot(annot, 'Artefact') #save the artefact events to the annotations file
        else:                                       # otherwise import noise scores from output of PSA.exe
            noisefile = rdir[0] + 'noise_scores.xls' 
            if path.isfile(noisefile):
                noisefile = pd.ExcelFile(noisefile).parse('Auto Scores') 
                for i, cn in enumerate(chan):
                    try:
                        noise = noisefile[[f'{cn}']].values
                        artefacts = empty((len(noise),8),dtype=object) #rearrange the data for importing into annotations file
                        x=-1
                        for i,val in enumerate(noise):
                            if val == 1:
                                x=x+1
                                artefacts[x,0] = x
                                artefacts[x,1] = i*5 
                                artefacts[x,2] = (artefacts[x,1]+(5))
                                #artefacts[x,3] = []
                                artefacts[x,7] = cn + ' (eeg)'        
                        artefacts = artefacts[0:x,:]
                        afile = root_dir + subj  + r'/wonambi//' + subj + f'-artefact_scores_{cn}.csv'     #save the rearranged events to csv    
                        savetxt(afile, artefacts, fmt="%s", delimiter=',', 
                                header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                        a = graphoelement.events_from_csv(afile) 
                        backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                                       str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                        shutil.copy(annot_file, backup_file) 
                        a.to_annot(annot, 'Artefact')       #save the artefact events to the annotations file  
                    except Exception as e:
                        print(f'WARNING: No artefact noise scores for {cn}.')
                        print('Check the electrodes for this recording and try again.')
                        continue
            else:
                print(f"Noise score file doesn't exist for {rdir[0]}")
    
    
    """ IMPORT AROUSALS AND SAVE TO ANNOTATIONS FILE """
    for r,rdir in enumerate(rec_dir):
        aroufile = rdir[0] + subj + '_arousals.txt'
        if path.isfile(aroufile):
            print(f'Importing arousals from {aroufile}')
            if system == 'alice':
                with open(aroufile, 'r') as f:
                    firstline = f.readline()
                if '#' in firstline:
                    arou = pd.read_csv(aroufile, sep="|", header=None, engine='python', skiprows=[0,1])
                    alice = 1  
                elif 'Alice Sleepware' in firstline:
                    arou = pd.read_csv(aroufile, sep="|", header=None, engine='python', skiprows=[0,1,2])
                    alice = 2
                else:
                    arou = pd.read_csv(aroufile, sep=None, header=None, engine='python')
                    alice = 3
            else:
                arou = pd.read_csv(aroufile, sep=None, header=None, engine='python')
        elif path.isfile(rdir[0] + subj + '_arousals.csv'):
            arou = pd.read_csv(rdir[0] + subj + '_arousals.csv', sep=None, header=None, engine='python')  
            firstline = arou[0,]
            if system == 'alice':
                if '#' in firstline:
                    arou = pd.read_csv(aroufile, sep="|", header=None, engine='python', skiprows=[0,1])
                    alice = 1  
                elif 'Alice Sleepware' in firstline:
                    arou = pd.read_csv(aroufile, sep="|", header=None, engine='python', skiprows=[0,1,2])
                    alice = 2
                else:
                    arou = pd.read_csv(aroufile, sep=None, header=None, engine='python')
                    alice = 3
        elif system == 'egi': 
            aroufile = [file for file in listdir(rdir[0]) if file.endswith("ScoredEvents.txt") if not file.startswith(".")][0]
        else:
            print(f"Arousals file doesn't exist for {rdir[0]}") 
            
        if system == 'alice':
            if alice == 1:
                arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
                dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
                i=-1
                for j,val in enumerate(arou[1]):
                    if arou[1].iloc[j] is not None:
                        if "rousal" in arou[1].iloc[j]:
                            i+=1
                            arousals[i,0] = i
                            start_date = dataset.header['start_time'].date()
                            arou_date = datetime.strptime(arou[3].iloc[j].split(" ")[1], "%d/%m/%Y").date()
                            
############### add_tancy begin ###################                           
                            if start_date == arou_date:
                                arousals[i,1] = (datetime.strptime(((arou[3].iloc[j].split(" ")[1]) + " " +(arou[3].iloc[j].split(" ")[2]) + " PM"), "%d/%m/%Y %I:%M:%S %p") - dataset.header['start_time']).seconds
                            else:
                                arousals[i,1] = (datetime.strptime(((arou[3].iloc[j].split(" ")[1]) + " " +(arou[3].iloc[j].split(" ")[2]) + " AM"), "%d/%m/%Y %I:%M:%S %p") - dataset.header['start_time']).seconds
                            
#                             try: arousals[i,1] = (datetime.strptime(arou[3].iloc[j].strip() + 'M', "%d/%m/%Y %I:%M:%S %p") - 
#                                              dataset.header['start_time']).seconds
#                             except: arousals[i,1] = (datetime.strptime(arou[3].iloc[j].strip(), "%d/%m/%Y %I:%M:%S %p") - 
#                                              dataset.header['start_time']).seconds 
                            
                            if not str(arou[4].iloc[j]) is None:
                                try: 
                                    ss = datetime.strptime(str(arou[4].iloc[j].strip()),"%S")
                                    ss = timedelta(minutes=ss.minute, seconds=ss.second)
                                    ss = ss.seconds
                                except:
                                    ss = datetime.strptime(str(arou[4].iloc[j].strip()),"%S.%f")
                                    ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                                    ss = ss.seconds + (ss.microseconds/1000000)
                                               

                            arousals[i,2] = arousals[i,1] + ss
                            arousals[i,7] = '(eeg)'        
                arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
                savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                        header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                a = graphoelement.events_from_csv(arfile)
                backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                               str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                shutil.copy(annot_file, backup_file)  
                a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file   
            elif alice == 2:
                arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
                dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
                i=-1
                for j,val in enumerate(arou[1]):
                    if arou[1].iloc[j] is not None:
                        if "rousal" in arou[1].iloc[j]:
                            i+=1
                            arousals[i,0] = i
                            if arou[3].iloc[j] is not None and 'A' in arou[3].iloc[j]:
                                arousals[i,1] = (datetime.strptime(arou[3].iloc[j].strip() + 'M', "%d/%m/%Y %I:%M:%S %p") - 
                                                 dataset.header['start_time']).seconds
                            elif arou[3].iloc[j] is not None:
                                
############### add_tancy begin ###################
                                if (datetime.strptime(arou[3][j].strip(), "%d/%m/%Y %H:%M:%S") - datetime.strptime(arou[3].iloc[0].strip(), "%d/%m/%Y %H:%M:%S")).days > 0:
                                    arousals[i,1] = (datetime.strptime(arou[3].iloc[j].strip() + ' AM', "%d/%m/%Y %I:%M:%S %p") - 
                                                 dataset.header['start_time']).seconds
                                else:
                                    arousals[i,1] = (datetime.strptime(arou[3].iloc[j].strip() + ' PM', "%d/%m/%Y %I:%M:%S %p") - 
                                                 dataset.header['start_time']).seconds
                                    
                            try: 
                                ss = datetime.strptime(str(arou[4].iloc[j].strip()),"%S")
                                ss = timedelta(minutes=ss.minute, seconds=ss.second)
                                ss = ss.seconds
                            except: 
                                ss = datetime.strptime(str(arou[4].iloc[j].strip()),"%S.%f")
                                ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                                ss = ss.seconds + (ss.microseconds/1000000) 
                            
#                             try: 
#                                 ss = datetime.strptime(str(arou[4].iloc[j+1]),"%S")
#                                 ss = timedelta(minutes=ss.minute, seconds=ss.second)
#                                 ss = ss.seconds
#                             except: 
#                                 ss = datetime.strptime(str(arou[4].iloc[j+1]),"%S.%f")
#                                 ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
#                                 ss = ss.seconds + (ss.microseconds/1000000)

############### add_tancy end ###################


                            arousals[i,2] = arousals[i,1] + ss
                            arousals[i,7] = '(eeg)'        
                arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
                savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                        header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                a = graphoelement.events_from_csv(arfile) 
                backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                               str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                shutil.copy(annot_file, backup_file)
                a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file
                
            elif alice == 3:
                arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
                dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
                i=-1
                for j,val in enumerate(arou[2].iloc[1:]):
                        if "rousal" in arou[0].iloc[j+1]:
                            i+=1
                            arousals[i,0] = i
                            arousals[i,1] = (datetime.strptime(dateStr + ' ' + arou[2].iloc[j+1], "%Y-%m-%d %I:%M:%S %p") - 
                                             dataset.header['start_time']).seconds
                            try: 
                                ss = datetime.strptime(str(arou[4].iloc[j+1]),"%S")
                                ss = timedelta(minutes=ss.minute, seconds=ss.second)
                                ss = ss.seconds
                            except: 
                                ss = datetime.strptime(str(arou[4].iloc[j+1]),"%S.%f")
                                ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                                ss = ss.seconds + (ss.microseconds/1000000)
                            
                            arousals[i,2] = arousals[i,1] + ss
                            arousals[i,7] = '(eeg)'        
                arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
                savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                        header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                a = graphoelement.events_from_csv(arfile) 
                backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                               str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                shutil.copy(annot_file, backup_file)    
                a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file
            
        elif system == 'compumedics' or system == 'grael':
            arou = pd.read_csv(aroufile, header=None, engine='python')
            arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
            dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
            for i,val in enumerate(arou[0]):
                    arousals[i,0] = i

############### add_tancy begin ###################
                    
                    arousals[i,1] = (datetime.strptime(dateStr + ' ' + arou[0][i], "%Y-%m-%d %H:%M:%S") - 
                                     dataset.header['start_time']).seconds
############### add_tancy end ###################

                                        
                    if "." in arou[4][i]:
                            ss = datetime.strptime(arou[4][i],"%M:%S.%f") 
                            ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                            ss = ss.seconds + (ss.microseconds/1000000)
                    else:
                            ss = datetime.strptime(arou[4][i],"%M:%S")
                            ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                            ss = ss.seconds
                    arousals[i,2] = arousals[i,1] + ss
                    arousals[i,7] = '(eeg)'        
            arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
            savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                    header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
            a = graphoelement.events_from_csv(arfile) 
            backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                           str(datetime.time(datetime.now())).replace(":", "_")[0:8])
            shutil.copy(annot_file, backup_file)
            a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file

        elif system == 'egi':                     
            aroufile = [file for file in listdir(rdir[0]) if file.endswith("ScoredEvents.txt") if not file.startswith(".")][0]
            if path.isfile(rdir[0] + aroufile):
                arou = pd.read_csv(rdir[0] + aroufile, sep=None, header=None, engine='python')
                arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
                dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
                i=-1
                for j,val in enumerate(noise[2][1:]):
                        if "rousal" in noise[2][j+1]:
                            i+=1
                            arousals[i,0] = i
                            arousals[i,1] = noise[0][j+1]
                            arousals[i,2] = float(arousals[i,1]) + float(noise[1][j+1])
                            arousals[i,7] = '(eeg)'        
                arfile = rdir[0]  + r'/wonambi//' + subj + '-arousals_scores.csv'     #save the rearranged events to csv    
                savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                        header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
                a = graphoelement.events_from_csv(arfile) 
                backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                               str(datetime.time(datetime.now())).replace(":", "_")[0:8])
                shutil.copy(annot_file, backup_file)
                a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file
                
        elif system == 'sandman':
            arou = pd.read_csv(aroufile, header=None, engine='python')
            arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
            dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
            for i,val in enumerate(arou[0][0:]):
                if 'Epoch' in arou[0][i]:
                    startline = i
            arou = arou[:][startline:]    
            for i,val in enumerate(arou[0][1:-1]):
                    arousals[i,0] = i+1
############### add_tancy begin ###################
                    
                    cur_time = arou[0].iloc[i+1].split("\t")[-2]
                    
                    arousals[i,1] = (datetime.strptime(dateStr + ' ' + cur_time, 
                            "%Y-%m-%d %I:%M:%S %p") - dataset.header['start_time']).seconds
                    
                    #arousals[i,1] = (datetime.strptime(dateStr + ' ' + arou[0].iloc[i+1][-16:-5].strip(), 
                    #        "%Y-%m-%d %I:%M:%S %p") - dataset.header['start_time']).seconds

############### add_tancy end ###################

                    if "." in arou[0].iloc[1+1][-5:]:
                            ss = datetime.strptime(arou[0].iloc[i+1][-5:].strip(),"%S.%f") 
                            ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                            ss = ss.seconds + (ss.microseconds/1000000)
                    else:
                            ss = datetime.strptime(arou[0].iloc[i+1][-5:].strip(),"%M:%S")
                            ss = timedelta(minutes=ss.minute, seconds=ss.second, microseconds=ss.microsecond)
                            ss = ss.seconds
                    arousals[i,2] = arousals[i,1] + ss
                    arousals[i,7] = '(eeg)'        
            arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
            savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                    header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
            a = graphoelement.events_from_csv(arfile) 
            backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                           str(datetime.time(datetime.now())).replace(":", "_")[0:8])
            shutil.copy(annot_file, backup_file)
            a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file
            
        elif system == 'remlogic':
            arou = pd.read_csv(aroufile, header=None, engine='python')[0]
            first_line = [l for l in arou if 'Time [hh:mm:ss]' in l]
            idx_first_line = arou==first_line[0]
            idx = [i for i, x in enumerate(idx_first_line) if x][0]
            arou = arou[idx+1:]
            arou = arou.reset_index(drop=True)
            arousals = empty((len(arou),8),dtype=object) #rearrange the data for importing into annotations file
            dateStr = dataset.header['start_time'].strftime("%Y-%m-%d")
            for i,val in enumerate(arou):

############### add_tancy begin ###################
                line = arou[i].split('\t')
                    
                arousals[i,0] = i+1
                if 'PM' in line[0]:
                    arousals[i,1] = (datetime.strptime(dateStr + ' ' + line[0][:-3], 
                        "%Y-%m-%d %I:%M:%S") + timedelta(hours = 12) - dataset.header['start_time']).seconds


                elif 'AM' in line[0]:
                    arousals[i,1] = (datetime.strptime(dateStr + ' ' + line[0][:-3], 
                        "%Y-%m-%d %I:%M:%S") - dataset.header['start_time']).seconds

#                     arousals[i,1] = (datetime.strptime(dateStr + ' ' + line[0][:-3], 
#                             "%Y-%m-%d %I:%M:%S %p") - dataset.header['start_time']).seconds


############### add_tancy end ###################
   
                    if "." in line[2]:
                            ss = datetime.strptime(line[2],"%S.%f") 
                    else:
                            ss = datetime.strptime(line[2],"%S")
                    ss = timedelta(minutes=ss.minute, seconds=ss.second)
                    ss = ss.seconds
                    arousals[i,2] = arousals[i,1] + ss
                    arousals[i,7] = '(eeg)'        
            arfile = rdir[0]  + r'/wonambi//' + subj + '-arousal_scores.csv'     #save the rearranged events to csv    
            savetxt(arfile, arousals, fmt="%s", delimiter=',', 
                    header ='Index,Start time,End time,Stitches,Stage,Cycle,Event type,Channel')
            a = graphoelement.events_from_csv(arfile) 
            backup_file = (backup[r][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                           str(datetime.time(datetime.now())).replace(":", "_")[0:8])
            shutil.copy(annot_file, backup_file)
            a.to_annot(annot, 'Arousal') #save the artefact events to the annotations file

def extract_ma(subj_list, root_dir, out_file, chan, ref_chan, nbins, rater, 
               cycle_idx, cat, stage, buffer, idpac, fpha, famp, filtcycle, 
               width, min_dur, evt_name ='slowwave', polar = 'normal', grp_name='eeg', 
               reject_artf = ['Artefact', 'Arousal'], laplacian='OFF', ch_names = None,
               coord_file = None, dcomplex = 'hilbert'):
    #
    # PAC script to exctract mean amplitudes of nested frequency per phase bin
    # for two fixed bands, 
    # on concatenated stage signal, 
    # with a Laplacian filter applied.
    #
    
    # arrange depending on whether concatenating cycles
    if cycle_idx is None:
        cycle = [None] 
    else: 
        cycle = cycle_idx

    if cat[0] == 1:
        cycle = [cycle]
    
    sg = ''.join(str(elem) for elem in stage)  
               
    all_ampbin = zeros((len(subj_list), len(chan), len(cycle)), dtype='object')
    
    
    # For single recording per subject
    for s, subj in enumerate(subj_list):
        if str(subj) != 'nan':
            rec_dir = root_dir + str(subj) # records folder
            xml_dir = root_dir + str(subj) + r'/wonambi//'
            r = [s for s in listdir(rec_dir) if s.endswith('.edf') or s.endswith('.set') if not s.startswith(".")]
            x = [s for s in listdir(xml_dir) if s.endswith('.xml') if not s.startswith(".")]
            records = [(r, x) for r, x in zip(r, x)]
            
            if path.exists(xml_dir + r'/cfc//'):
                print(xml_dir + r'/cfc//' + " exists")
            else:
                mkdir((xml_dir + r'/cfc//'))
                print(xml_dir + r'/cfc//' + " created")
            out = xml_dir + r'/cfc//' + out_file + '.p'
            
    # FOR Repeated Measures (to be completed)
    # ### read records from directories, alphanumerically
    # records = [(r, x) for r,x in zip(sorted(listdir(rec_dir), key=str.lower), 
    #             sorted(listdir(xml_dir), key=str.lower))]
    
    # # sort in experimental and control nights
    # memo = [x for x in records if x[0][10] == 'm']
    # ctrl = [x for x in records if x[0][10] == 'c']
    # records = [(c, m) for c, m in zip(ctrl, memo)]       
  
            ### open dataset
            dset = Dataset(rec_dir + '/' + records[0][0])
            
            ### check polarity of recording
            if isinstance(polar, list):
                polarity = polar[s]
            else:
                polarity = 'normal'
            ### import Annotations file, select rater
            annot = Annotations(xml_dir + '/' + records[0][1], rater_name=rater)
            
            ### get cycles
            if cycle_idx is not None:
                all_cycles = annot.get_cycles()
                cycle = [all_cycles[i - 1] for i in cycle_idx if i <= len(all_cycles)]
                outsub_ampbin = zeros((len(chan), len(cycle)*nbins))
            else:
                cycle = [None]
                outsub_ampbin = zeros((len(chan), nbins))
            
            # create subject amp bin object
            sub_ampbin = zeros((1, len(chan), len(cycle)), dtype='object')
            chanrow = []
            # run through channels
            for k, ch in enumerate(chan):
                chan_full = ch + ' (' + grp_name + ')'
                if ch == '_REF':
                    ch = 'Cz'
                chanrow.append(ch)
                # run through cycles    
                for l, cyc in enumerate(cycle):
                    if not isinstance(cyc, list):
                        cyc = [cyc]
                    
                    if cat[0] == 1 and cat[1] == 1 :
                        print('Reading data for ' + records[0][0] + ', channel ' + ch + ', stages ' + sg + ' , all cycles')
                    elif cat[1] == 1:
                        print('Reading data for ' + records[0][0] + ', channel ' + ch + ', stage ' + sg + ', cycle ' + str(l + 1))
                    elif cat[1] == 0 and len(stage)>1:
                        print('***')
                        print('ERROR! Please check concatenation and concatenate stages.')
                        print('To get estimates per stage, run stages separately.')
                        print('***')
                        return
                     
                    ### select and read data    
                    segments = fetch(dset, annot, cat=cat, chan_full=[chan_full], evt_type=[evt_name],
                                     cycle=cyc, stage=stage, buffer=buffer,
                                     reject_epoch=True, reject_artf=reject_artf)
                    if laplacian == 'ON':
                        segments.read_data(chan=ch_names, ref_chan=ref_chan)
                    else:
                        segments.read_data(ch, ref_chan=ref_chan, grp_name=grp_name)
                    
                    ampbin = zeros((len(segments), nbins))
                    
                    ### Define PAC object
                    pac = Pac(idpac=idpac, f_pha=fpha, f_amp=famp, dcomplex=dcomplex, 
                              cycle=filtcycle, width=width, n_bins=nbins)
                    print('***')
                    print('Length segments = {}'.format(len(segments)))
                    print('(IF 0 PLEASE CHECK EVENTS ARE PRESENT IN XML).')
                    print('***')
                    for m, seg in enumerate(segments):
                        print('Calculating mean amplitudes for segment {} of {}, {}, {}'.format(m + 1, 
                              len(segments), records[0][0], ch))
                        data = seg['data']
                        s_freq = data.s_freq
                        if laplacian == 'ON':
                            data = surface_laplacian(data, ch_names, coord_file,
                                                     return_chan=[ch])[0, :]
                        else:
                            data = seg['data'].data[0][0]
                        
                        if polarity == 'normal':
                            None
                        elif polarity == 'opposite':
                            data = data*-1    
                        
                        pha = pac.filter(s_freq, data, ftype='phase',n_jobs=1)
                        amp = pac.filter(s_freq, data, ftype='amplitude',n_jobs=1)
                        nbuff = int(buffer * s_freq)
                        minlen = s_freq * min_dur
                        if len(pha) >= 2 * nbuff + minlen:
                            pha = pha[nbuff:-nbuff]
                            amp = amp[nbuff:-nbuff]
                        ampbin[m, :] = mean_amp(pha, amp, nbins=nbins)                                          
            
                    # create output summaries
                    outsub_ampbin[k,l*nbins:l*nbins+nbins] = mean(ampbin,axis=0)
                    sub_ampbin[0, k, l] = ampbin
                    all_ampbin[s, k, l] = ampbin
            
            
            # save mean amps to output file per subject        
            with open(out, 'wb') as f:
                dump(sub_ampbin, f)
            print(chanrow)    
            df1 = pd.DataFrame(chanrow)
            df2 = pd.DataFrame(outsub_ampbin)
            mat = pd.concat([df1, df2], axis=1)
            savetxt(xml_dir + r'/cfc//' + out_file + ".csv", mat, fmt="%s", delimiter=',')
            
    # Save all subject ma's to group-level file    
    if path.exists(root_dir + r'/group//'):
        print(root_dir + r'/group//' + " exists")
    else:
        mkdir((root_dir + r'/group//'))
        print(root_dir + r'/group//' + " created")        
    if path.exists(root_dir + r'/group/cfc//'):
        print(root_dir + r'/group//cfc//' + " exists")
    else:
        mkdir((root_dir + r'/group/cfc//'))
        print(root_dir + r'/group/cfc//' + " created")
    out = root_dir + r'/group/cfc//' + out_file + '.p'                        
    with open(out, 'wb') as f:
                dump(all_ampbin, f)

def plot_ma(root_dir, subj_list, chan, stage, nbins, bp_greek, band_pairs=None):
    # Plot nightwise mean amplitudes per bin and preferred phase
    nsubs = len(subj_list)
    numplots = len(chan)
    sg = ''.join(str(elem) for elem in stage)
    cg = ''.join(str(elem) for elem in chan)
    ab_list = []  
    cfc_dir = root_dir + '/group/cfc//'
    ab_list = [s for s in listdir(cfc_dir) if (".p") in s]
    ab_list = [x for x in ab_list if not x.startswith('.')]
    ab_list = [x for x in ab_list if sg in x]
    ab_list = [x for x in ab_list if cg in x]
    if band_pairs is None:
        band_pairs = [s.split('_')[0] for s in ab_list if (".p") in s]
    bp = len(band_pairs)            
    pph = empty((nsubs,numplots))
    

    for i, bp in enumerate(band_pairs):  
        shift = -int(nbins / 4) # correction for phase shift introduced by Hilbert trf
        vecbin = zeros(nbins)
        width = 2 * pi / nbins
        for n in range(nbins):
            vecbin[n] = n * width + width / 2
        norm = True
        f_c, axarr_c = plt.subplots(numplots, 1, sharex=True, sharey='row', squeeze=False)
        f_c.tight_layout(pad=2.0)
    
        #xL_c = [r'- $\pi$', r'- $\pi$/2', '0', r'$\pi$/2', r'$\pi$']
        xL_c = ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2', r'2$\pi$']
        #x = linspace(-pi, pi, 100)
        x = linspace(0, 2*pi, 100)
        
        f_p, axarr_p = plt.subplots(numplots, 1, subplot_kw=dict(projection='polar'), squeeze=False)
        xL_p = ['0', '', '', '', r'+/- $\pi$', '', '', ''] 
        
        ab_file = cfc_dir + ab_list[i]
        print(f'Saving... {ab_file}')
        with open(ab_file, 'rb') as f:
            ab = load(f) # ab[i,:,:,:] = #sub; ab[:,i,:,:] = #chan; ab[:,:,i,:] = #cyc; ab[:,:,:,i] = #bin;
            ab = _allnight_ampbin(ab, 0, nbins, norm=norm)
            ab = roll(ab, shift, axis=-1) # IMPORTANT ~@#!
        for j in range(ab.shape[1]):
            xab = nanmean(ab[:, j, :], axis=0)
            one_ax = axarr_c[j % numplots, j // numplots]
            one_ax.plot(x, sin(x) * std(xab) + mean(xab), '.85')
            one_ax.plot(vecbin, xab)
            #one_ax.set_xticks([-pi, -pi/2, 0, pi/2, pi])
            one_ax.set_xticks([0, pi/2, pi, 3 * pi / 2, 2 * pi])
            one_ax.set_xticklabels(xL_c)
            one_ax.set_title(chan[j] + '-' + bp)
            newab = ab[:, j, :]
            mask = all(isnan(newab), axis=1)
            newab = newab[~mask]
            for s, subj in enumerate(subj_list):
                pph[s,j] = vecbin[newab[s].argmax(axis=-1)]
            w1, bin_edges = histogram(pph[:,j], bins=nbins, range=(0, 2*pi))
            oneax = axarr_p[j % numplots, j // numplots]
            oneax.bar(bin_edges[:-1], w1, width=width, bottom=0.0)
            oneax.set_xticklabels(xL_p)
            oneax.set_yticks([])
            oneax.set_title(chan[j] + '-' + stage[0] + '-' + bp,loc='left')

            for r, row in enumerate(bp_greek):
                one_ax.set_ylabel(row, labelpad=20, rotation=0, size='small')
                oneax.set_ylabel(row, labelpad=60, rotation=0, size='small')
        
       
        ppp = reshape(pph,(nsubs,numplots))
        ppp_degrees = empty((nsubs,numplots))
        for s in range(len(subj_list)):
            ppp_degrees[s,:] = [math.degrees(x) for x in ppp[s]]
        subs = ['chan'] + subj_list
        subs = reshape(transpose(subs),(len(subs),1))
        chanhead = [cc+'_radians' for cc in chan]
        ppp = vstack((chanhead,ppp))
        chanhead = [cc+'_degrees' for cc in chan]
        ppp_degrees = vstack((chanhead,ppp_degrees))
        ppp = hstack((subs,ppp,ppp_degrees))
        
        # Save pref phase to group-level file            
        if path.exists(root_dir + r'/group/cfc/figs//'):
            print(root_dir + r'/group/cfc/figs//' + " exists")
        else:
            mkdir((root_dir + r'/group/cfc/figs//'))
            print(root_dir + r'/group/cfc/figs//' + " created")
        fig_dir = root_dir + r'/group/cfc/figs//'
        f_c.savefig(fig_dir + f'{bp}_ampbins_{sg}_{cg}.png')    
        f_p.savefig(fig_dir + f'{bp}_preferred_phase_{sg}_{cg}.png') 
        #savetxt(root_dir + r'/group/cfc//' + f"{bp}_preferred_phase_{sg}_{cg}.csv", ppp, fmt="%s", delimiter=',')

def plot_ma_per_subject(root_dir, subj_list, chan, stage, nbins, bp_greek, band_pairs=None):
    # Plot nightwise mean amplitudes per bin and preferred phase
    subs = len(subj_list)
    numplots = len(chan)
    ab_list = []  
    sg = ''.join(str(elem) for elem in stage)
    cg = ''.join(str(elem) for elem in chan)
    # For single recording per subject
    for s, subj in enumerate(subj_list):
        if str(subj) != 'nan':
            cfc_dir = root_dir + str(subj) + '/wonambi/cfc//'
            ab_list = [s for s in listdir(cfc_dir) if (".p") in s]
            ab_list = [x for x in ab_list if not x.startswith('.')]
            ab_list = [x for x in ab_list if sg in x]
            ab_list = [x for x in ab_list if cg in x]
            if len(ab_list) == 0:
                print(f'ERROR. No valid files found with {sg}, {cg} for subject {subj}')
                continue
            if band_pairs is None:
                band_pairs = [s.split('_')[0] for s in ab_list if (".p") in s]
            bp = len(band_pairs)            
            pph = empty((1,numplots,bp), dtype='object')
            ppp = empty((1,numplots,bp), dtype='object')
            ppa = empty((1,numplots,bp), dtype='object')
            

            
            for i, bp in enumerate(band_pairs):
                
                ### no need to modify these variables
                shift = -int(nbins / 4) # correction for phase shift introduced by Hilbert trf
                vecbin = zeros(nbins)
                width = 2 * pi / nbins
                for n in range(nbins):
                    vecbin[n] = n * width + width / 2
                norm = True
                
                f_c, axarr_c = plt.subplots(numplots, 1, sharex=True, sharey='row', squeeze=False)
                f_c.tight_layout(pad=2.0)
            
                xL_c = ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2', r'2$\pi$']
                x = linspace(0, 2*pi, 100)
                
                f_p, axarr_p = plt.subplots(numplots, 1, subplot_kw=dict(projection='polar'), squeeze=False)
                xL_p = ['0', '', '', '', r'+/- $\pi$', '', '', ''] 
                
                f_pa, axarr_pa = plt.subplots(numplots, 1, subplot_kw=dict(projection='polar'), squeeze=False)
                
                
                ab_file = cfc_dir + ab_list[i]
                print(ab_file)
                with open(ab_file, 'rb') as f:
                    ab = load(f) #ab[i,:,:] = #chan; ab[:,i,:] = #stage; ab[:,:,i] = #cycle; inside last index is ampbins
                    ab2 = ab[:]
                    ab = _allnight_ampbin(ab, 0, nbins, norm=norm)
                    ab = roll(ab, shift, axis=-1) # IMPORTANT ~@#!
                    
                for j in range(ab.shape[1]):
                    
                    # Amplitudes across phase bins for all night
                    xab = nanmean(ab[:, j, :], axis=0)
                    one_ax = axarr_c[j % numplots, j // numplots]
                    one_ax.plot(x, sin(x) * std(xab) + mean(xab), '.85')
                    one_ax.plot(vecbin, xab)
                    #one_ax.set_xticks([-pi, -pi/2, 0, pi/2, pi])
                    one_ax.set_xticks([0, pi/2, pi, 3 * pi / 2, 2 * pi])
                    one_ax.set_xticklabels(xL_c)
                    one_ax.set_title(chan[j] + '-' + bp)
                    
                    # Preferred phase across
                    newab2 = ab2[:, j, 0]
                    newab2 = roll(newab2[0][:],shift, axis=-1)
                    pph[0,j,i] = vecbin[newab2.argmax(axis=-1)]
                    w1, bin_edges = histogram(pph[0,j,:][0], bins=nbins, range=(0, 2*pi))
                    oneax = axarr_p[j % numplots, j // numplots]
                    oneax.bar(bin_edges[:-1], w1, width=width, bottom=0.0)
                    oneax.set_xticklabels(xL_p)
                    oneax.set_yticks([])
                    oneax.set_title(chan[j] + '-' + stage[0] + '-' + bp,loc='left')
                    
                    newab = ab[:, j, :]
                    mask = all(isnan(newab), axis=1)
                    newab = newab[~mask]
                    ppa[0,j,i] = vecbin[newab.argmax(axis=-1)]
                    w1, bin_edges = histogram(ppa[0,j,:][0], bins=nbins, range=(0, 2*pi))
                    oneax = axarr_pa[j % numplots, j // numplots]
                    oneax.bar(bin_edges[:-1], w1, width=width, bottom=0.0)
                    oneax.set_xticklabels(xL_p)
                    oneax.set_yticks([])
                    oneax.set_title(chan[j] + '-' + stage[0] + '-' + bp,loc='left')
            
            for i, row in enumerate(bp_greek):
                one_ax.set_ylabel(row, labelpad=20, rotation=0, size='small')
                oneax.set_ylabel(row, labelpad=60, rotation=0, size='small')
        
            ppp = reshape(ppa,(numplots*len(band_pairs),1))
            ppp_degrees = [math.degrees(x) for x in ppp]
            ppp_degrees = reshape(ppp_degrees,(numplots*len(band_pairs),1))
            ppp = vstack((chan[0]+'radians',ppp))
            ppp_degrees = vstack((chan[0]+'degrees',ppp_degrees))
            ppp = hstack((ppp,ppp_degrees))
    
            
            if path.exists(cfc_dir + r'/figs//'):
                print(cfc_dir + '/figs/ already exists')
            else:
                mkdir(cfc_dir + r'/figs//')
            fig_dir = cfc_dir + r'/figs//'
            f_c.savefig(fig_dir + f'{bp}_ampbins_{sg}_{cg}.png')    
            f_p.savefig(fig_dir + f'{bp}_preferred_phase_{sg}_{cg}.png') 
                
                
            savetxt(cfc_dir + f"{bp}_preferred_phase_{sg}_{cg}.csv", ppp, fmt="%s", delimiter=',')        
    
def calculate_mi(root_dir, subj_list, chan, stage, nbins, band_pairs=None):

    subs = len(subj_list)
    numplots = len(chan)
    all_mi = []
    ab_list = []
    sg = ''.join(str(elem) for elem in stage)
    cg = ''.join(str(elem) for elem in chan)
    
    # Determine the band pairs to analyse
    cfc_dir = root_dir + r'/group/cfc//'
    ab_list = [s for s in listdir(cfc_dir) if (".p") in s]
    ab_list = [x for x in ab_list if not x.startswith('.')]
    ab_list = [x for x in ab_list if not x.startswith('.')]
    ab_list = [x for x in ab_list if sg in x]
    ab_list = [x for x in ab_list if cg in x]
    if band_pairs is None:
                band_pairs = [s.split('_')[0] for s in ab_list if (".p") in s]
    
    for i, bp in enumerate(band_pairs):
    
        # For single recording per subject
        for s, subj in enumerate(subj_list):
            if str(subj) != 'nan':
                cfc_dir = root_dir + str(subj) + '/wonambi/cfc//'
        
                ### no need to modify these variables
                shift = -int(nbins / 4) # correction for phase shift introduced by Hilbert trf        
                data_row = [bp]
                ab_file = cfc_dir + ab_list[i]
                with open(ab_file, 'rb') as f:
                    ab = load(f)
                ab = _allnight_ampbin(ab, 0, nbins=nbins, norm=True)
                ab = roll(ab, shift, axis=-1)
                
                mi = klentropy(ab)
                all_mi.append(mi)
                lmi = log(mi)
                all_mi.append(lmi)
                hd = f"{chan}"
                mi = reshape(mi, (-1,1))
                lmi = reshape(lmi, (-1,1))
                mi = [format_float_positional(m) for m in mi]        
                lmi = [format_float_positional(lm) for lm in lmi] 
                midata = [hd, mi, lmi]
                
                lmifile = cfc_dir + f"/{bp}_lmi_{nbins}bin_buffer_{sg}_{cg}.csv"  #save the rearranged events to csv    
                savetxt(lmifile, midata, delimiter=',',fmt='%s')
                    
                    
        all_mi = reshape(all_mi, (subs,numplots*len(band_pairs)*2))
        #all_mi = (log(all_mi))
        michan = [c + '_' + bp + "_mi" for c in chan for bp in band_pairs]
        lmichan = [c + '_' + bp + "_logmi" for c in chan for bp in band_pairs]
        all_mi = vstack((hstack((michan,lmichan)),all_mi))
        subs = ['chan'] + subj_list
        subs = reshape(transpose(subs),(len(subs),1))
        all_mi = hstack((subs, all_mi))
        
        allmifile = root_dir + r'/group/cfc//' + f"/{bp}_lmi_{nbins}bin_buffer_{sg}_{cg}.csv" 
        savetxt(allmifile, all_mi, delimiter=',',fmt='%s')
    
    return all_mi

def expand_events(subj_list, root_dir, evt_type, rater='Anon', chan_grp_name='eeg'):
               
    for s, subj in enumerate(subj_list):
        if str(subj) != 'nan':
            rec_dir = root_dir + 'individual/' + str(subj) # records folder
            xml_dir = root_dir + 'individual/' + str(subj) + r'/wonambi//'
            r = [s for s in listdir(rec_dir) if (".edf") in s]
            x = [s for s in listdir(xml_dir) if (".xml") in s]
            records = [(r, x) for r, x in zip(r, x)]

            ### Open dataset
            dset = Dataset(rec_dir + '/' + records[0][0])
                        
            ### Import Annotations file, select rater
            annot_file = xml_dir + '/' + records[0][1]
            annot = Annotations(annot_file, rater_name=rater)
            
            if path.exists(root_dir + 'individual/' + subj + r'/wonambi//'):
                        print(root_dir + 'individual/' + subj + " already exists")
            else:
                    mkdir((root_dir + 'individual/' + subj + r'/wonambi//'))
                    mkdir((root_dir + 'individual/' + subj + r'/wonambi/backups//'))
                  
            rec_dir = [[root_dir + 'individual/' + subj + r'/']] # records folder
            backup = [[root_dir + 'individual/' + subj + r'/wonambi/backups//']] #backup folder
            
            
            backup_file = (backup[0][0] + subj + '-' + str(datetime.date(datetime.now()))+ '_' + 
                       str(datetime.time(datetime.now())).replace(":", "_")[0:8] + '.xml')
            shutil.copy(annot_file, backup_file) 
        
            ### Specify event channels
            chans = dset.header['chan_name']
            chans_full = None
            if chans:
                chans_full = [i + ' (' + chan_grp_name + ')' for i in chans]
            
            ### Read events from annotations file and rename channel
            print('Reading data for ' + records[0][0])
            for c,chan_full in enumerate(chans_full):
                events = annot.get_events(evt_type,chan=chan_full)
                if len(events) > 0:
                    for e,evt in enumerate(events):
                        evt['chan'] = ['']
                    grapho = graphoelement.Graphoelement()
                    grapho.events = events
                    grapho.to_annot(annot, evt_type) #save the artefact events to the annotations file
                    
def whale_it(in_dir, method, chan, rater, cat, stage, ref_chan, grp_name, 
             frequency=(11, 16), duration= (0.5, 3), polar = 'normal', part='all', visit='all'):
    
    # First we check the output directory
    for p,pt in enumerate(part):
        if str(pt) != 'nan':
            # a. check for wonambi folder, if doesn't exist, skip
            if not path.exists(in_dir + '/' + pt + r'/wonambi//'):
                    print(f'WARNING. No wonambi folder for {pt}. Skipping...')
            else: 
                
                backup = in_dir + '/' + pt+ r'/wonambi/' + r'backups/'
                
                # loop through records
                if isinstance(part, list):
                    None
                elif part == 'all':
                        part = listdir(in_dir)
                        part = [ p for p in part if not(p.startswith('.'))]
                else:
                    print("ERROR: 'part' must either be an array of subject ids or = 'all' ")       
                
                print(r"""Whaling it... (searching for spindles)
                                      .
                                   ":"
                                 ___:____     |"\/"|
                               ,'        `.    \  /
                               |  O        \___/  |
                             ~^~^~^~^~^~^~^~^~^~^~^~^~
                             """)    
                
                
                ## Define files
                rec_dir = in_dir + '/' + pt + '/'
                ant_dir = rec_dir + '/wonambi/'
                edf_file = [x for x in listdir(rec_dir) if x.endswith('.edf') or x.endswith('.set') or x.endswith('.rec') or x.endswith('.eeg') if not x.startswith('.')]
                xml_file = [x for x in listdir(ant_dir) if x.endswith('.xml') if not x.startswith('.')]            
                
                ## Copy annotations file before beginning
                
                backup_file = backup + pt + '-' + str(datetime.date(datetime.now()))+ '_' + str(datetime.time(datetime.now())).replace(":", "_")[0:8] + '.xml'
                                   
                shutil.copy(ant_dir + xml_file[0], backup_file)
                
                ## Now import data
                dset = Dataset(rec_dir + edf_file[0])
                annot = Annotations(ant_dir + xml_file[0], rater_name=rater)
                ### check polarity of recording
                if isinstance(polar, list):
                    polarity = polar[p]
                else:
                    polarity = 'normal'
                ## Select and read data
                for c, ch in enumerate(chan):
                    try:
                        print(f'Reading data for {pt}, ' + str(ch))
                        segments = fetch(dset, annot, cat=cat, stage=stage, cycle=None, 
                                         reject_epoch=True, reject_artf=['Artefact', 'Arou', 'Arousal'])
                        segments.read_data(ch, ref_chan, grp_name=grp_name)
                    
                    
                        ## Loop through methods (i.e. WHALE IT!)
        
                        for m, meth in enumerate(method):
                            print(meth)
                            ### define detection
                            detection = DetectSpindle(meth, frequency=frequency, duration=duration)
                                
                            ### run detection and save to Annotations file
                            all_spin = []
                            
                            for i, seg in enumerate(segments):
                                print('Detecting events, segment {} of {}'.format(i + 1, 
                                      len(segments)))
                                if polarity == 'normal':
                                    None
                                elif polarity == 'opposite':
                                    seg['data'].data[0][0] = seg['data'].data[0][0]*-1
                                spindles = detection(seg['data'])
                                spindles.to_annot(annot, 'spindle')
                                all_spin.append(spindles)
                    except:
                        print(f'WARNING: NO {ch} for {pt}, Skipping... ')
    ## all_spin contains some basic spindle characteistics
    print('Detection complete and saved.')        
    return

def whale_farm(in_dir, system_list, method, chan, grp_name, rater, stage=None, 
               reject_artf = ['Artefact', 'Arou', 'Arousal'], ref_chan=[], 
               evt_name=['spindle'], cycle_idx=None, frequency=(11, 16),
               part='all', param_keys=None, exclude_poor=False, epoch_dur=30, n_fft_sec=4):
    
    logfile = in_dir + '/' + 'whales_log.txt'
    log = open(logfile, "a")
    
    print('')
    print(f'{datetime.now()}')
    print(f'Running whale_farm on directory {in_dir}')  
    print('')
    with open(logfile, 'a') as f:
                    print(f'Running whale_farm on directory {in_dir}', file=f) 
                    

    # Get list of participants, recording systems - and sort them
    if isinstance(part, list):
        None
    elif part == 'all':
            part = listdir(in_dir)
            part = [ p for p in part if not(p.startswith('.'))]
    else:
        print("ERROR: 'part' must either be an array of subject ids or = 'all' **CHECK**")
        with open(logfile, 'a') as f:
            print('', file=f) 
            print("ERROR: 'part' must either be an array of subject ids or = 'all' ", file=f)  
    
    try:
        indices = [i for i,s in enumerate(part) if type(s) is float if isnan(s)]
        part.remove(nan)
        [system_list.pop(i) for i in indices]
    except:
        None
    
    print(f'Participants = {part}')
    part_system = concat([DataFrame(part), DataFrame(system_list)], axis=1)
    part_system.columns= ['part','system']
    part_system = part_system.sort_values(by=['part'])
    part = list(part_system['part'])
    system_list = list(part_system['system'])
    
    
    # Set base parameters
    params = {}
    sublist = []
    header = []
    
    if param_keys is None:
        param_keys = ['count','density','dur', 'ptp', 'energy', 'peakef'] # Default spindle metrics to extract
    if cycle_idx is not None:
            for m, param in enumerate(param_keys):
                params[param] = zeros((len(part),len(chan)*len(cycle_idx)*len(stage)+len(chan)+len(stage)))
                params[param].fill(nan)
    else:
            for m, param in enumerate(param_keys):
                params[param] = zeros((len(part),len(chan)*len(stage)+len(chan)))
                params[param].fill(nan) 
    

    # Loop through subjects          
    for i, p in enumerate(part):
        out_dir = in_dir + '/' + p + '/wonambi/'
        dat = []
        sublist.append(p)

        
        # Update size of file based on number of visits(sessions)
        if i == 0:
            for m, param in enumerate(param_keys):
                    params[param] = zeros((len(part), len(params[param][0])))
                    params[param].fill(nan)
        
  
        if not path.exists(out_dir):   
            print(f'WARNING: no Wonambi folder for Subject {p}, skipping..')
            with open(logfile, 'a') as f:
                print('', file=f) 
                print(f'WARNING: no Wonambi folder for Subject {p}, skipping..', file=f)  
            
    
        # Define files
        rec_dir = in_dir + '/' + p + '/' 
        edf_file = [x for x in listdir(rec_dir) if x.endswith('.edf') or x.endswith('.set') or x.endswith('.rec') or x.endswith('.eeg') if not x.startswith('.')]
        xml_file = [x for x in listdir(in_dir + p + r'/wonambi/' ) if x.endswith('.xml') if not x.startswith('.')] 
        
        # Open dataset
        dataset = Dataset(rec_dir + edf_file[0])

        # Import Annotations file
        annot = Annotations(rec_dir + r'wonambi/' + xml_file[0], 
                            rater_name=rater)
    
        # Get sleep cycles (if any)
        if cycle_idx is not None:
            all_cycles = annot.get_cycles()
            cycle = [all_cycles[y - 1] for y in cycle_idx if y <= len(all_cycles)]
        else:
            cycle = None
        
        
        # Run through channels
        for ch, channel in enumerate(chan):
            chan_ful = [channel + ' (' + grp_name + ')']
            
            # Create header for output file
            for m, param in enumerate(param_keys):
                if i == 0:
                    header.append(param + channel + '_wholenight' )
            for s, st in enumerate(stage):
                for m, param in enumerate(param_keys):
                                if i == 0:
                                    header.append(param + '_' + channel + '_' + st )
                if cycle_idx is not None:
                    for cy, cyc in enumerate(cycle_idx):
                        for m, param in enumerate(param_keys):
                                if i == 0:
                                    header.append(param + '_' + channel + '_' + st + '_cycle' + str(cy+1) )
            
            ### WHOLE NIGHT ###
            # Select and read data
            segments = []
            data = []
            #try:
            print('Reading data for ' + p + ', ' + channel)
            with open(logfile, 'a') as f:
                print('', file=f) 
                print('Reading data for ' + p + ', ' + channel, file=f)
            
            segments = fetch(dataset, annot, cat=(0,0,0,0), evt_type=[evt_name], cycle=cycle, 
                                chan_full=chan_ful, reject_epoch=False, 
                                reject_artf = reject_artf, min_dur=0)
            
            segments.read_data(channel, ref_chan, grp_name=grp_name)
                
            # except:
            #     print('')
            #     print(f"WARNING: {channel} doesn't exist for {p}, skipping...")
            #     print('')
            #     dat.append(nan)
            #     for m, param in enumerate(param_keys[1:]):
            #         dat.append(nan)
            #     with open(logfile, 'a') as f:
            #         print('', file=f) 
            #         print(f"WARNING: {channel} doesn't exist for {p}, skipping...", file=f)

            #     continue
            
            if len(segments) == 0:
                print('')
                print(f"WARNING: {evt_name}'s haven't been detected for {p} on channel {channel}, skipping...")
                print('')
                dat.append(nan)
                for m, param in enumerate(param_keys[1:]):
                    dat.append(nan)
                with open(logfile, 'a') as f:
                    print('', file=f) 
                    print("WARNING: {evt_name}'s haven't been detected for {p} on channel {channel}, skipping...", file=f)
                    
                for s, st in enumerate(stage):
                    dat.append(nan)
                    for m, param in enumerate(param_keys[1:]):
                        dat.append(nan)
                    if cycle_idx is not None: 
                        for cy, cycc in enumerate(cycle_idx):
                            dat.append(nan)
                            for m, param in enumerate(param_keys[1:]):
                                dat.append(nan)
                    
            else:        
                if isinstance(chan_ful, list):
                            if len(chan_ful) > 1:
                                chan_ful = chan_ful[0] 
                elif not isinstance(chan_ful, list):
                    chan_ful = [chan_ful]
            
                # Calculate event density (whole night)
                poi = get_times(annot, stage=stage, cycle=cycle, chan=[channel], exclude=exclude_poor)
                total_dur = sum([x[1] - x[0] for y in poi for x in y['times']])
                evts = annot.get_events(name=evt_name, chan = chan_ful, stage = stage)
                count = len(evts)
                density = len(evts) / (total_dur / epoch_dur)
                print('')
                print('----- WHOLE NIGHT -----')
                print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}')
                print(f'Density = {density} per min')
                print('')
                with open(logfile, 'a') as f:
                    print('', file=f) 
                    print('----- WHOLE NIGHT -----', file=f) 
                    print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}', file=f) 
                    print(f'Density = {density} per min', file=f) 
                col = int(((len(params['density'][i])/len(chan))*ch))
                dat.append(count)
                dat.append(density)
                
                # Set n_fft
                n_fft = None
                if segments and n_fft_sec is not None:
                    s_freq = segments[0]['data'].s_freq
                    n_fft = int(n_fft_sec * s_freq)
                
                # Export event parameters (whole night)
                data = event_params(segments, params='all', band=frequency, n_fft=n_fft)
               
                if data:
                        data = sorted(data, key=lambda x: x['start'])
                        outputfile = out_dir + p + '_' + channel + '_' + evt_name + '.csv'
                        print('Writing to ' + outputfile)
                        with open(logfile, 'a') as f:
                            print('', file=f) 
                            print('Writing to ' + outputfile, file=f) 
                        export_event_params(outputfile, data, count=len(evts), 
                                            density=density)
                else:
                    print('No valid data found.')
                    with open(logfile, 'a') as f:
                            print('', file=f) 
                            print('No valid data found.', file=f) 
                    
                for ev in data:
                    ev['ptp'] = ev['ptp']()[0][0]
                    ev['energy'] = list(ev['energy'].values())[0]
                    ev['peakef'] = list(ev['peakef'].values())[0]
                    #ev['minamp'] = data[0]['minamp'].data[0][0]
                    #ev['maxamp'] = data[0]['maxamp'].data[0][0]
                    #ev['rms'] = data[0]['rms'].data[0][0]
                
                for m, param in enumerate(param_keys[2:]):
                    col = int(((len(params['density'][i])/len(chan))*ch))
                    dat.append(asarray([x[param] for x in data]).mean())
                
                ### PER STAGE ###
                for s, st in enumerate(stage):
                    data = []
                    if isinstance(chan_ful, list):
                            if len(chan_ful) > 1:
                                chan_ful = chan_ful[0]
                    elif not isinstance(chan_ful, list):
                        chan_ful = [chan_ful]
                    
                    try:
                        segments = fetch(dataset, annot, cat=(0,0,0,0), evt_type=[evt_name], 
                                         stage = [st], cycle=cycle, 
                                         chan_full=chan_ful, reject_epoch=True, 
                                         reject_artf = reject_artf, min_dur=0.5)
                        segments.read_data(channel, ref_chan, grp_name=grp_name)
                    except:
                        print('')
                        print(f"WARNING: {channel} doesn't exist for {p}, skipping...")
                        print('')
                        dat.append(nan)
                        for m, param in enumerate(param_keys[2:]):
                            dat.append(nan)
                        with open(logfile, 'a') as f:
                            print('', file=f) 
                            print(f"WARNING: {channel} doesn't exist for {p}, skipping...", file=f)
        
                        continue
            
                    if len(segments) == 0:
                        print('')
                        print(f"WARNING: {evt_name}'s haven't been detected for {p} on channel {channel} in {st}, skipping...")
                        print('')
                        dat.append(nan)
                        for m, param in enumerate(param_keys[2:]):
                            dat.append(nan)
                        with open(logfile, 'a') as f:
                            print('', file=f) 
                            print("WARNING: {evt_name}'s haven't been detected for {p} on channel {channel} in {st}, skipping...", file=f)
                    else:        
                        if len(chan_ful) > 1:
                                chan_ful = chan_ful[0]
        
                        # Calculate event density (per stage)
                        poi = get_times(annot, stage=[st], cycle=cycle, chan=[channel], exclude=exclude_poor)
                        total_dur = sum([x[1] - x[0] for y in poi for x in y['times']])
                        evts = annot.get_events(name=evt_name, chan = chan_ful, stage = st)
                        count = len(evts)
                        density = len(evts) / (total_dur / epoch_dur)
                        print('')
                        print(f'---- STAGE {st} ----')
                        print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}')
                        print(f'Density = {density} per min')
                        print('')
                        with open(logfile, 'a') as f:
                                print('', file=f)
                                print(f'---- STAGE {st} ----', file=f)
                                print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}', file=f)
                                print(f'Density = {density} per min', file=f)
                        dat.append(count)
                        dat.append(density)

                        
                        # Set n_fft
                        n_fft = None
                        if segments and n_fft_sec is not None:
                            s_freq = segments[0]['data'].s_freq
                            n_fft = int(n_fft_sec * s_freq)
                        
                        # Export event parameters (per stage)
                        data = event_params(segments, params='all', band=frequency, n_fft=n_fft)
                            
                        for ev in data:
                            ev['ptp'] = ev['ptp']()[0][0]
                            ev['energy'] = list(ev['energy'].values())[0]
                            ev['peakef'] = list(ev['peakef'].values())[0]
            
                        
                        for m, param in enumerate(param_keys[2:]):
                            dat.append(asarray([x[param] for x in data]).mean())
                        
                        ### PER CYCLE ###
                        if cycle_idx is not None: 
                                    try:
                                        for cy, cycc in enumerate(cycle_idx):
                                            data = []
                                            if not isinstance(chan_ful, list):
                                                chan_ful = [chan_ful]
                                                if len(chan_ful) > 1:
                                                    chan_ful = chan_ful[0] 
                                            segments = fetch(dataset, annot, cat=(0,0,0,0), evt_type=[evt_name], 
                                                             stage = [st], cycle=[cyc], 
                                                             chan_full=chan_ful, reject_epoch=True, 
                                                             reject_artf = reject_artf, min_dur=0.5)
                    
                                        
                                            segments.read_data([channel], ref_chan, grp_name=grp_name)
                                            
                                            if isinstance(chan_ful, ndarray):
                                                if len(chan_ful) > 1:
                                                    chan_ful = chan_ful[0]
                
                                            if len(chan_ful) > 1:
                                                chan_ful = chan_ful[0]
                
                                            # Calculate event density (per cycle)
                                            poi = get_times(annot, stage=[st], cycle=[cyc], chan=[channel], exclude=exclude_poor)
                                            total_dur = sum([x[1] - x[0] for y in poi for x in y['times']])
                                            count = len(evts)
                                            evts = annot.get_events(name=evt_name, time=cycle[cy][0:2], chan = chan_ful, stage = st)
                                            density = len(evts) / (total_dur / epoch_dur)
                                            print('')
                                            print(f'---- STAGE {st}, CYCLE {cy+1} ----')
                                            print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}')
                                            print(f'Density = {density} per min')
                                            print('')
                                            with open(logfile, 'a') as f:
                                                print('', file=f)
                                                print(f'---- STAGE {st}, CYCLE {cy+1} ----', file=f)
                                                print(f'No. Segments = {len(segments)}, Total duration (s) = {total_dur}', file=f)
                                                print(f'Density = {density} per min', file=f)
                                            dat.append(count)
                                            dat.append(density)

                                            
                                            # Set n_fft
                                            n_fft = None
                                            if segments and n_fft_sec is not None:
                                                s_freq = segments[0]['data'].s_freq
                                                n_fft = int(n_fft_sec * s_freq)
                                            
                                            # Export event parameters (per cycle)
                                            data = event_params(segments, params='all', band=frequency, n_fft=n_fft)
                                                
                                            for ev in data:
                                                ev['ptp'] = ev['ptp']()[0][0]
                                                ev['energy'] = list(ev['energy'].values())[0]
                                                ev['peakef'] = list(ev['peakef'].values())[0]
                                                           
                                            for m, param in enumerate(param_keys[2:]):
                                                dat.append(asarray([x[param] for x in data]).mean())
                                                
            
                                    except Exception:
                                        print(f'No STAGE {st} in CYCLE {cy+1}')
                                        dat.append(nan)
                                        for m, param in enumerate(param_keys[2:]):
                                            dat.append(nan)
    
                        
        
        if i == 0:
            output = DataFrame(dat)
        else:
            output = concat([output,DataFrame(dat)], axis=1)
                
        
        #filler = empty((len(header)-output.shape[0],output.shape[1]))
        #filler.fill(nan)
        #output = concat([output,DataFrame(filler)], axis=1) #concatenate((output, filler))
        #output = DataFrame(output)
    output = DataFrame.transpose(output) 
    output.columns=header  
    output.index=sublist                 
    
    # Create naming labels for output files
    systems = ''
    for element in list(set(system_list)):
        systems += str(element) + '_'
    
    channames = ''
    for element in list(set(chan)):
        channames += str(element) + '_'
    
    # Save outputfile
    DataFrame.to_csv(output, f'{in_dir}/{evt_name}_{channames}{systems}dataset.csv', sep=',', na_rep='NaN')            
    with open(f'{out_dir}/{evt_name}.p', 'wb') as f:
        dump(params, f)
            
        return output
        
def slow_it(in_dir, method, chan, rater, cat, stage, ref_chan, grp_name, 
            reject_artf=['Artefact', 'Arou', 'Arousal'],
             frequency=(0.1,4), duration= [], polar = 'normal', #duration= (0.3, 1)
             part='all', visit='all'):
    
    # First we set up the output directory
    # a. check for derivatives folder, if doesn't exist, create

    
    # loop through records
    if isinstance(part, list):
        None
    elif part == 'all':
            part = listdir(in_dir)
            part = [ p for p in part if not(p.startswith('.'))]
    else:
        print("ERROR: 'part' must either be an array of subject ids or = 'all' ")       
    
    print(r"""Detecting slow waves... 
                          .
``'-.,_,.-'``'-.,_,.='``'-.,_,.-'``'-.,_,.='``
                 """)    
    
    for i, p in enumerate(part):
        if str(p) != 'nan':
            # a. check for wonambi folder, if doesn't exist, skip
            if not path.exists(in_dir + '/' + p + r'/wonambi//'):
                    print(f'WARNING. No wonambi folder for {p}. Skipping...')
            else: 
                    backup = in_dir + '/' + p + r'/wonambi/' + r'backups/'

            rec_dir = in_dir + '/' + p + '/' 

            annot_dir = in_dir + '/' + p + r'/wonambi//'

            #edf_file = [x for x in listdir(rec_dir) if x.endswith('.edf') or x.endswith('.rec') or x.endswith('.eeg') if not x.startswith('.')]
            

            edf_file = [x for x in listdir(rec_dir) if x.endswith('.edf') or x.endswith('.set') or x.endswith('.rec') or x.endswith('.eeg') if not x.startswith('.')]
            xml_file = [x for x in listdir(annot_dir) if x.endswith('.xml') if not x.startswith('.')]            
            
            ## Copy annotations file before beginning
            backup_file = backup + p + '-' + str(datetime.date(datetime.now()))+ '_' + str(datetime.time(datetime.now())).replace(":", "_")[0:8] + '.xml'

            shutil.copy(annot_dir + xml_file[0], backup_file)
            
            ## Now import data
            dset = Dataset(rec_dir + edf_file[0])
            annot = Annotations(annot_dir + xml_file[0], rater_name=rater)
            ### check polarity of recording
            if isinstance(polar, list):
                polarity = polar[p]
            else:
                polarity = 'normal'
            ## Select and read data
        
            try:
                print(f'Reading data for {p}, all channels')
                segments = []
                segments = fetch(dset, annot, cat=cat, stage=stage, cycle=None, 
                                 reject_epoch=True, reject_artf=reject_artf)
                segments.read_data(chan, ref_chan, grp_name=grp_name)
                
                ## Loop through methods (i.e. WHALE IT!)
    
                for m, meth in enumerate(method):
                    print(meth)
                    ### define detection
                    detection = DetectSlowWave(meth, frequency=frequency, duration=duration)
                        

                    ### create output csv file for slow wave model fit results
                    det_name = str(detection).replace("/", "_") # avoid saving dir error
                    csv_name = p + '_' + ch + '_slowwave_' + det_name + '.csv'
                    csv_out = path.join(annot_dir, csv_name)

                    ### run detection and save to Annotations file
                    swaves = []
                    for i, seg in enumerate(segments):
                        print('Detecting events, segment {} of {}'.format(i + 1, 
                              len(segments)))
                        if polarity == 'normal':
                            None
                        elif polarity == 'opposite':
                            seg['data'].data[0][0] = seg['data'].data[0][0]*-1 

                        swaves = detection(seg['data'])
                        swaves.to_annot(annot, 'slowwave')
                    
                # export detecion output as csv
                keys = swaves.events[0].keys()                       
                with open(csv_out, 'w', newline='') as output_file:    
                    dict_writer = csv.DictWriter(output_file, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(swaves.events)
                print('\nData exported to csv succesfully')
        
            except: 
                print(f'All channels not available for {p}') 
                for c, ch in enumerate(chan):
                    try:
                        print(f'Reading data for {p}, ' + str(ch))
                        segments = []
                        segments = fetch(dset, annot, cat=cat, stage=stage, cycle=None, 
                                         reject_epoch=True, reject_artf=reject_artf)
                        segments.read_data(ch, ref_chan, grp_name=grp_name)
                        ## Loop through methods (i.e. WHALE IT!)
            
                        for m, meth in enumerate(method):
                            print(meth)
                            ### define detection
                            detection = DetectSlowWave(meth, duration=duration)
                            
                            ### create output csv file for slow wave model fit results
                            det_name = str(detection).replace("/", "_") # avoid saving dir error
                            csv_name = p + '_' + ch + '_slowwave_' + det_name + '.csv'
                            csv_out = path.join(annot_dir, csv_name)


                            ### run detection and save to Annotations file
                            swaves = []
                            for i, seg in enumerate(segments):
                                print('Detecting events, segment {} of {}'.format(i + 1, 
                                      len(segments)))
                                if polarity == 'normal':
                                    None
                                elif polarity == 'opposite':
                                    seg['data'].data[0][0] = seg['data'].data[0][0]*-1 
        
                                swaves = detection(seg['data'])
                                swaves.to_annot(annot, 'slowwave')
                                
                        # export detecion output as csv
                        keys = swaves.events[0].keys()                       
                        with open(csv_out, 'w', newline='') as output_file:    
                            dict_writer = csv.DictWriter(output_file, keys)
                            dict_writer.writeheader()
                            dict_writer.writerows(swaves.events)
                        print('\nData exported to csv succesfully')

                    except:
                        print(f'WARNING: NO {ch} for {p}, Skipping... ')
                    
    ## all_spin contains some basic spindle characteistics
    print('Detection complete and saved.')        
    return




def replace_outliers(in_dir, chan, rater=None, evt_name=['slowwave'], part='all'):

    if path.exists(in_dir):
        print('')
        print(f'Time start: {datetime.now()}')
        print(f'Replacing outliers from files in directory {in_dir}')  
        print(f'Event = {evt_name[0]}')
        print('')
        
        # Loop through subjects
        if isinstance(part, list):
            None
        elif part == 'all':
                part = listdir(in_dir)
                part = [ p for p in part if not(p.startswith('.')) if path.isdir(in_dir)]
        else:
            print("ERROR: 'part' must either be an array of subject ids or = 'all' **CHECK**")
        
        
        part.sort()
        for i, p in enumerate(part):
            # Loop through visits
    
                if path.exists(in_dir + p + r'/wonambi//'):
        
                    # Define files
                    xml_dir = in_dir + p + r'/wonambi//' 
                    xml_file = [x for x in listdir(xml_dir) if x.endswith('.xml') if not x.startswith('.')] 
                    
                    #evt_dir = xml_dir +  r'/slowwave/'  ### CHANGE HERE FROM slowwave to real folder name
                    evt_dir = xml_dir + r'/Clean/'  ### CHANGE HERE FROM slowwave to real folder name
                    print(evt_dir)
                    
                    if len(xml_file) == 0:                
                        print(f'WARNING: No XML for Subject {p}, skipping..')
                        
                    else:
                        # backup file
                        backup_file = (p + '-' + str(datetime.date(datetime.now()))+ '_' + 
                                       str(datetime.time(datetime.now())).replace(":", "_")[0:8] +'.xml')
                        shutil.copy(xml_dir + xml_file[0], xml_dir + backup_file) 
                        
                        # Import Annotations file
                        annot = Annotations(xml_dir + xml_file[0], 
                                            rater_name=rater)
                        annot.remove_event_type(name=evt_name[0])
                        
                        # Run through channels
                        for ch, channel in enumerate(chan):
                            # Select and read data
                            print('Reading data for ' + p + ', visit ' + ' ' + channel)
                            csvfile = [x for x in listdir(evt_dir) if x.endswith('.csv') if not x.startswith('.') 
                                       if channel in x ]
                            print(csvfile)
                            evt_file = evt_dir + csvfile[0]
                            grapho = graphoelement.Graphoelement()
                            evts = graphoelement.events_from_csv(evt_file)   
                            grapho.events = evts
                            grapho.to_annot(annot, 'slowwave')
                else:
                    print("ERROR:" + in_dir + r'/wonambi//' + " doesn't exist")

    else:
        print("ERROR:" + in_dir + " doesn't exist")
    return