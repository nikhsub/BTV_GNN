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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000))

ClusteringParam = cms.PSet(
          seedMax3DIPSignificance =  cms.double(9999.0),
          seedMax3DIPValue =  cms.double(9999.0),
          seedMin3DIPSignificance = cms.double(1.2),    # 1.2 sigma    
          seedMin3DIPValue = cms.double(0.005),         # 50um
          
          clusterMaxDistance = cms.double(0.05),
          clusterMaxSignificance = cms.double(4.5),
          distanceRatio = cms.double(20.0),
          clusterMinAngleCosine = cms.double(0.5),
          maxTimeSignificance = cms.double(99999), #3.5 from Nik
)

arbitrator = cms.PSet(
        primaryVertices       = cms.InputTag("dummy"),
        secondaryVertices     = cms.InputTag("dummy"),           # unused by your call
        tracks                = cms.InputTag("dummy"),           # unused by your call
        beamSpot              = cms.InputTag("dummy"),

        dRCut                 = cms.double(0.40),
        distCut               = cms.double(0.040),
        sigCut                = cms.double(5.0),
        dLenFraction          = cms.double(0.333),
        fitterSigmacut        = cms.double(3.0),
        fitterTini            = cms.double(256.0),
        fitterRatio           = cms.double(0.25),
        trackMinLayers        = cms.int32(4),
        trackMinPt            = cms.double(0.4),
        trackMinPixels        = cms.int32(1),
        maxTimeSignificance   = cms.double(3.0)
    )


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/04A0B676-D63A-6D41-B47F-F4CF8CBE7DB8.root") #Training
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0C6623EF-B101-694A-8904-D7578B1093C8.root") #Testing
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0925253B-38AF-CE43-978A-476DF939963D.root") #Training
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/0381415E-3E93-1546-835C-4C73CD974231.root") #Testing
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/4938DD1E-2F40-0D4B-ACD4-39D2D98F25BE.root") #Testing post
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/9D704B6A-B565-5A43-B5DD-B73A5A014582.root")
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/B86867C3-7DC0-1045-80E5-9F060A4B0547.root")
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/04A0B676-D63A-6D41-B47F-F4CF8CBE7DB8.root")
    #fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18MiniAODv2/HPlusCharm_3FS_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2550000/042288EE-8A62-1742-955D-FD759DE1B8AF.root")
)

process.mergedGenParticles = cms.EDProducer("MergedGenParticleProducer",
                                            inputPruned = cms.InputTag("prunedGenParticles"),
                                            inputPacked = cms.InputTag("packedGenParticles"),
                                            )


process.demo = cms.EDAnalyzer('DemoAnalyzer',
    packed = cms.InputTag("packedGenParticles"),
    pruned = cms.InputTag("prunedGenParticles"),
    beamspot = cms.untracked.InputTag('offlineBeamSpot'), # GC
    merged = cms.InputTag("mergedGenParticles"),
    tracks = cms.untracked.InputTag('packedPFCandidates'),
    jets = cms.untracked.InputTag('slimmedJets'),
    primaryVertices = cms.untracked.InputTag('offlineSlimmedPrimaryVertices'),
    secVertices = cms.untracked.InputTag('slimmedSecondaryVertices'),
    losttracks = cms.untracked.InputTag('lostTracks', '', "PAT"),
    TrackPtCut = cms.untracked.double(0.5),
    addPileupInfo = cms.untracked.InputTag('slimmedAddPileupInfo'),
    vtxweight = cms.untracked.double(0.5),
    vertexfitter = cms.untracked.PSet(
             finder = cms.string('avr')
         ),
    TrackPredCut = cms.untracked.double(0.0), # set to 0 should be IVF normal reconstruction?
    clusterizer = ClusteringParam,
    model_path = cms.FileInPath("dispV/dispVAnalyzer/data/focloss_out48_hadtrain_2606.onnx")
    #genmatch_csv = cms.FileInPath("dispV/dispVAnalyzer/data/geninfo_ntup_ttbarhad_2807.csv")
)
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("ttbar_hadronic_2807_genvertexing.root"),
)


#inclusiveVertexFinder  = cms.EDProducer("InclusiveVertexFinder",
#       beamSpot = cms.InputTag("offlineBeamSpot"),
#       primaryVertices = cms.InputTag('offlineSlimmedPrimaryVertices'),
#       tracks = cms.InputTag("generalTracks"),
#       minHits = cms.uint32(8),
#       maximumLongitudinalImpactParameter = cms.double(0.3),
#       minPt = cms.double(0.8),
#       maxNTracks = cms.uint32(30),
#
#       clusterizer = cms.PSet(
#           seedMax3DIPSignificance = cms.double(9999.),
#           seedMax3DIPValue = cms.double(9999.),
#           seedMin3DIPSignificance = cms.double(1.2),
#           seedMin3DIPValue = cms.double(0.005),
#           clusterMaxDistance = cms.double(0.05), #500um
#           clusterMaxSignificance = cms.double(4.5), #4.5 sigma
#           distanceRatio = cms.double(20), # was cluster scale = 1 / density factor =0.05 
#           clusterMinAngleCosine = cms.double(0.5), # only forward decays
#       ),
#
#       vertexMinAngleCosine = cms.double(0.95), # scalar prod direction of tracks and flight dir
#       vertexMinDLen2DSig = cms.double(2.5), #2.5 sigma
#       vertexMinDLenSig = cms.double(0.5), #0.5 sigma
#       fitterSigmacut =  cms.double(3),
#       fitterTini = cms.double(256),
#       fitterRatio = cms.double(0.25),
#       useDirectVertexFitter = cms.bool(True),
#       useVertexReco  = cms.bool(True),
#       vertexReco = cms.PSet(
#               finder = cms.string('avr'),
#               primcut = cms.double(1.0),
#               seccut = cms.double(3),
#               smoothing = cms.bool(True)
#       )
#

#)


process.p = cms.Path(process.mergedGenParticles + process.demo)
#process.p = cms.Path(process.inclusiveVertexFinder 

#+ producer.IVF

#)
