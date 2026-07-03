import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing("analysis")

options.register(
    "outfile",
    "out_ttbar.root",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Output training ROOT file",
)
options.register(
    "globalTag",
    "106X_upgrade2018_realistic_v16_L1v1",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "GlobalTag to use for graph-training production",
)
options.parseArguments()

process = cms.Process("GraphProduction")

process.load("Configuration.StandardSequences.Services_cff")
process.load("JetMETCorrections.Configuration.JetCorrectors_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.GlobalTag.globaltag = cms.string(options.globalTag)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

ClusteringParam = cms.PSet(
    seedMax3DIPSignificance=cms.double(9999.0),
    seedMax3DIPValue=cms.double(9999.0),
    seedMin3DIPSignificance=cms.double(0.0),
    seedMin3DIPValue=cms.double(0.00),
    clusterMaxDistance=cms.double(0.05),
    clusterMaxSignificance=cms.double(4.5),
    distanceRatio=cms.double(20.0),
    clusterMinAngleCosine=cms.double(0.0),
    maxTimeSignificance=cms.double(99999),
)

process.source = cms.Source("PoolSource", 
        #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/HPlusCharm_3FS_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2550000/00A8CDC7-44C1-0C4D-9EF8-098027DC2B47.root")
        fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0925253B-38AF-CE43-978A-476DF939963D.root") #Training
)

process.mergedGenParticles = cms.EDProducer(
    "MergedGenParticleProducer",
    inputPruned=cms.InputTag("prunedGenParticles"),
    inputPacked=cms.InputTag("packedGenParticles"),
)

process.demo = cms.EDAnalyzer(
    "TrainAnalyzer",
    training=cms.untracked.bool(True),
    packed=cms.InputTag("packedGenParticles"),
    pruned=cms.InputTag("prunedGenParticles"),
    beamspot=cms.untracked.InputTag("offlineBeamSpot"),
    merged=cms.InputTag("mergedGenParticles"),
    tracks=cms.untracked.InputTag("packedPFCandidates"),
    jets=cms.untracked.InputTag("slimmedJets"),
    primaryVertices=cms.untracked.InputTag("offlineSlimmedPrimaryVertices"),
    secVertices=cms.untracked.InputTag("slimmedSecondaryVertices"),
    losttracks=cms.untracked.InputTag("lostTracks", "", "PAT"),
    TrackPtCut=cms.untracked.double(0.5),
    addPileupInfo=cms.untracked.InputTag("slimmedAddPileupInfo"),
    vtxweight=cms.untracked.double(0.5),
    vertexfitter=cms.untracked.PSet(finder=cms.string("avr")),
    clusterizer=ClusteringParam,
    model_path=cms.FileInPath("dispV/dispVAnalyzer/data/bhive_hcmod_1703.onnx"),
)

process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outfile))

process.p = cms.Path(process.mergedGenParticles + process.demo)

