import CRABClient
import subprocess
from subprocess import getstatusoutput
#from commands import getstatusoutput
import sys, os
import argparse
from multiprocessing import Process

from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException
#from httplib import HTTPException
import datetime

#def getstatusoutput(cmd):
#    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    output, _ = process.communicate()
#    status = process.returncode
#    return (status, output.decode('utf-8'))

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%d%m")

config = config()
config.General.transferOutputs = True
config.General.transferLogs = False
config.General.workArea = "crab_projects"

config.JobType.pluginName = 'Analysis'
#config.JobType.maxMemoryMB = 3000
#config.JobType.allowUndistributedCMSSW = True

config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.totalUnits = 70

config.Data.outLFNDirBase = '/store/user/nvenkata/BTV/ttbarhad_70files_'+str(timestamp)
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
#config.Site.blacklist = ['T3_US_UCR']

from multiprocessing import Process
#config.Site.whitelist = ['T3_US_Colorado', 'T2_US_Florida', 'T3_CH_PSI', 'T2_DE_RWTH', 'T2_CH_CERN', 'T2_US_*', 'T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest', 'T2_IT_Rome', 'T2_IT_Bari', 'T2_IT_Legnaro', 'T2_FR_CCIN2P3', 'T2_FR_GRIF_LLR', 'T2_DE_DESY', 'T2_DE_RWTH', 'T2_UK_London_Brunel', 'T2_ES_CIEMAT', 'T2_ES_IFCA', 'T2_BE_IIHE']
config.Site.whitelist = ['T1_*', 'T2_*', 'T3_*']

#def produce_new_cfg(mass, life, lines):
#    file = open("XXTo4J/XXTo4J_M"+str(mass)+"_CTau"+str(life)+"mm_CP2_GENSIM.py", "w")
#    #width = 0.0197327e-11/float(life)
#    #print width
#    for line in lines:
#        newline = line.replace("AAAA", str(mass)).replace("BBBB", str(life))
#        file.write(newline)
#    file.close()

def submit(config):
    try:
        crabCommand('submit', config = config)
    except HTTPException as hte:
        print("Failed submitting task: %s", (hte.headers))
    except ClientException as cle:
        print("Failed submitting task: %s", (cle))


def sub_crab_job():

    #datasetname = getstatusoutput("das_client --query='dataset=/splitSUSY_M1000_"+str(mass)+"_ctau"+str(life)+"p0_TuneCP2_13TeV-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM*'")[1].split("\n")[0]
    datasetname = getstatusoutput("das_client --query='dataset='/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM*")[1].split("\n")[0]

    #datasetname = getstatusoutput("das_client --query='dataset='/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM*")[1].split("\n")[0]

   # dataset_query = "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM"
    dataset_query = "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM"
    files_query = f"file dataset={dataset_query}"
    status, files = getstatusoutput(f"dasgoclient --query='{files_query}'")
    
    if status != 0:
        print("Error querying DAS for files:", files)
        return
    
    # Filter files from 100 to 200 (zero-indexed)
    file_list = files.split('\n')[0:70]
    if not file_list:
        print("No files found in the specified range (0–70).")
        return

    # Save the filtered file list to a local text file (optional)
    with open("filelist_0_70.txt", "w") as f:
        for file in file_list:
            f.write(file + '\n')

    config.Data.userInputFiles = file_list
    config.General.requestName = 'MC_ttbarhad_'+str(timestamp)
    config.JobType.psetName = 'Events_cfg.py'
    config.Data.outputDatasetTag = 'MC_ttbarhad_'+str(timestamp)

    #config.Data.inputDataset = datasetname
    #print(datasetname)
    #submit(config)
    p = Process(target=submit, args=(config,))
    p.start()
    p.join()


#infile = open('XXTo4J_MAAAA_CTauBBBBmm_TuneCP2_13TeV_pythia8_GENSIM.py')
#lines = infile.readlines()
sub_crab_job()
