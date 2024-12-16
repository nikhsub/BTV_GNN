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
config.Data.totalUnits = 62
config.Data.outLFNDirBase = '/store/group/lpcljm/nvenkata/BTVH/ttbarlep_'+str(timestamp)
config.Data.publication = False

config.Site.storageSite = 'T3_US_FNALLPC'
#config.Site.blacklist = ['T3_US_UCR']

from multiprocessing import Process
#config.Site.whitelist = ['T3_US_Colorado', 'T2_US_Florida', 'T3_CH_PSI', 'T2_DE_RWTH']#['T2_CH_CERN', 'T2_US_*', 'T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest', 'T2_IT_Rome', 'T2_IT_Bari', 'T2_IT_Legnaro', 'T2_FR_CCIN2P3', 'T2_FR_GRIF_LLR', 'T2_DE_DESY', 'T2_DE_RWTH', 'T2_UK_London_Brunel', 'T2_ES_CIEMAT', 'T2_ES_IFCA', 'T2_BE_IIHE']

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
    #datasetname = getstatusoutput("das_client --query='dataset='/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM*")[1].split("\n")[0]
    datasetname = getstatusoutput("das_client --query='dataset='/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM*")[1].split("\n")[0]
    config.General.requestName = 'MC_ttbarlep_'+str(timestamp)
    config.JobType.psetName = 'Events_cfg.py'
    config.Data.outputDatasetTag = 'MC_ttbarlep_'+str(timestamp)
    config.Data.inputDataset = datasetname
    print(datasetname)
    #submit(config)
    p = Process(target=submit, args=(config,))
    p.start()
    p.join()


#infile = open('XXTo4J_MAAAA_CTauBBBBmm_TuneCP2_13TeV_pythia8_GENSIM.py')
#lines = infile.readlines()
sub_crab_job()
