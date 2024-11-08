import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration.StandardSequences.Services_cff')
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('RecoTracker.Configuration.RecoTracker_cff')
#process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.GlobalTag.globaltag =  cms.string("106X_upgrade2018_realistic_v16_L1v1")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(150))

process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/04A0B676-D63A-6D41-B47F-F4CF8CBE7DB8.root") #Training
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0C6623EF-B101-694A-8904-D7578B1093C8.root") #Testing
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0925253B-38AF-CE43-978A-476DF939963D.root") #Training
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0381415E-3E93-1546-835C-4C73CD974231.root") #Testing
)

process.mergedGenParticles = cms.EDProducer("MergedGenParticleProducer",
                                            inputPruned = cms.InputTag("prunedGenParticles"),
                                            inputPacked = cms.InputTag("packedGenParticles"),
                                            )


process.demo = cms.EDAnalyzer('DemoAnalyzer',
packed = cms.InputTag("packedGenParticles"),
pruned = cms.InputTag("prunedGenParticles"),
merged = cms.InputTag("mergedGenParticles"),
tracks = cms.untracked.InputTag('packedPFCandidates'),
jets = cms.untracked.InputTag('slimmedJets'),
primaryVertices = cms.untracked.InputTag('offlineSlimmedPrimaryVertices'),
secVertices = cms.untracked.InputTag('slimmedSecondaryVertices'),
losttracks = cms.untracked.InputTag('lostTracks', '', "PAT"),
TrackPtCut = cms.untracked.double(0.5),
addPileupInfo = cms.untracked.InputTag('slimmedAddPileupInfo')
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string("ttbar_had_48k_3110_train.root"),
)

process.p = cms.Path(process.mergedGenParticles+process.demo)
