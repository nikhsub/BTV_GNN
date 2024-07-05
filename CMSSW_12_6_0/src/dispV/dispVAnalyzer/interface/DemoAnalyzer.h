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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

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
      bool checkPDG(int abs_pdg);


      
      const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
      edm::EDGetTokenT<pat::PackedCandidateCollection> TrackCollT_;
      edm::EDGetTokenT<reco::VertexCollection> PVCollT_;
      edm::EDGetTokenT<pat::PackedCandidateCollection> LostTrackCollT_;
      edm::EDGetTokenT<edm::View<reco::Jet> > jet_collT_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > prunedGenToken_;
      edm::EDGetTokenT<edm::View<pat::PackedGenParticle> > packedGenToken_;
      
      TTree *tree;
      double TrackPtCut_;

      std::vector<float> Hadron_pt;
      std::vector<float> Hadron_eta;
      std::vector<float> Hadron_phi;
      std::vector<float> Hadron_SVx;
      std::vector<float> Hadron_SVy;
      std::vector<float> Hadron_SVz;
      std::vector<int> nHadrons;
      std::vector<int> nSV;
      std::vector<float> Daughter1_pt;
      std::vector<float> Daughter1_eta;
      std::vector<float> Daughter1_phi;
      std::vector<float> Daughter1_charge;
      std::vector<float> Daughter2_pt;
      std::vector<float> Daughter2_eta;
      std::vector<float> Daughter2_phi;
      std::vector<float> Daughter2_charge;

      std::vector<int> ntrks;
      std::vector<float> trk_ip2d;
      std::vector<float> trk_ip3d;
      std::vector<float> trk_ip2dsig;
      std::vector<float> trk_ip3dsig;
      std::vector<float> trk_pt;
      std::vector<float> trk_eta;
      std::vector<float> trk_phi;

      std::vector<int> njets;
      std::vector<float> jet_pt;
      std::vector<float> jet_eta;
      std::vector<float> jet_phi;

};

#endif // DemoAnalyzer_h
