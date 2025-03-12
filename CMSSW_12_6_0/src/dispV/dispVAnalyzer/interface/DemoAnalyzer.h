#ifndef DemoAnalyzer_h
#define DemoAnalyzer_h

// system include files
#include <memory>
#include <tuple>
#include <optional>
#include <limits>
#include <cmath>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

//TFile Service

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//ROOT
#include "TTree.h"

#include "math.h"

//Transient Track
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

//Pileup info
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//IPTOOLS
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
//
// class declaration
//
class DemoAnalyzer : public edm::one::EDAnalyzer<> {
   public:
      explicit DemoAnalyzer (const edm::ParameterSet&);
      ~DemoAnalyzer();
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
      
   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      std::optional<std::tuple<float, float, float>> isAncestor(const reco::Candidate * ancestor, const reco::Candidate * particle);
      int checkPDG(int abs_pdg);
      bool hasDescendantWithId(const reco::Candidate* particle, const std::vector<int>& pdgIds);

      
      const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
      edm::EDGetTokenT<pat::PackedCandidateCollection> TrackCollT_;
      edm::EDGetTokenT<reco::VertexCollection> PVCollT_;
      edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate>> SVCollT_;
      edm::EDGetTokenT<pat::PackedCandidateCollection> LostTrackCollT_;
      edm::EDGetTokenT<edm::View<reco::Jet> > jet_collT_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > prunedGenToken_;
      edm::EDGetTokenT<edm::View<pat::PackedGenParticle> > packedGenToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > mergedGenToken_;

      TTree *tree;
      double TrackPtCut_;

      edm::EDGetTokenT<std::vector<PileupSummaryInfo>> PupInfoT_;

      float nPU;

      std::vector<float> Hadron_pt;
      std::vector<float> Hadron_eta;
      std::vector<float> Hadron_phi;
      std::vector<float> Hadron_GVx;
      std::vector<float> Hadron_GVy;
      std::vector<float> Hadron_GVz;
      std::vector<int> nHadrons;
      std::vector<int> nGV;
      std::vector<int> nGV_B;
      std::vector<int> nGV_D;
      std::vector<int> GV_flag;
      std::vector<int> nDaughters;
      std::vector<int> nDaughters_B;
      std::vector<int> nDaughters_D;
      std::vector<int> Daughters_flag;
      std::vector<int> Daughters_flav;
      std::vector<float> Daughters_pt;
      std::vector<float> Daughters_eta;
      std::vector<float> Daughters_phi;
      std::vector<float> Daughters_charge;
           
      std::vector<int> ntrks;
      std::vector<float> trk_ip2d;
      std::vector<float> trk_ip3d;
      std::vector<float> trk_ip2dsig;
      std::vector<float> trk_ip3dsig;
      std::vector<float> trk_p;
      std::vector<float> trk_pt;
      std::vector<float> trk_eta;
      std::vector<float> trk_phi;
      std::vector<float> trk_charge;
      std::vector<int> trk_nValid;
      std::vector<int> trk_nValidPixel;
      std::vector<int> trk_nValidStrip;
	
      std::vector<int> trk_i;
      std::vector<int> trk_j;
      std::vector<float> deltaR;
      std::vector<float> dca;
      std::vector<float> rel_ip2d;
      std::vector<float> rel_ip3d;
      
      std::vector<int> njets;
      std::vector<float> jet_pt;
      std::vector<float> jet_eta;
      std::vector<float> jet_phi;

      std::vector<int> nSVs;
      std::vector<float> SV_x;
      std::vector<float> SV_y;
      std::vector<float> SV_z;
      std::vector<float> SV_pt;
      std::vector<float> SV_mass;
      std::vector<int> SV_ntrks;
      std::vector<float> SVtrk_pt;
      std::vector<float> SVtrk_eta;
      std::vector<float> SVtrk_phi; 

};

#endif // DemoAnalyzer_h
