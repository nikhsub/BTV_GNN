// -*- C++ -*-
//
// Package:    Demo/TrainAnalyzer
// Class:      TrainAnalyzer
//
/**\class TrainAnalyzer TrainAnalyzer.cc Demo/TrainAnalyzer/plugins/TrainAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Nikhilesh Venkatasubramanian
//         Created:  Tue, 16 Apr 2024 18:10:25 GMT
//
//

//
// constructors and destructor

#include "dispV/dispVAnalyzer/interface/TrainAnalyzer.h"
#include "dispV/dispVAnalyzer/interface/matchedHadronsToSV.h"
#include <iostream>
#include <omp.h>
#include <unordered_set>
#include <iomanip>
#include <unordered_map>
#include "TH1F.h"
#include <cmath>

 //function to get 3D distance bewtween all SV and all GV
static std::vector<std::vector<float>> computeDistanceMatrix(
                const std::vector<float>& SV_x,
                const std::vector<float>& SV_y,
                const std::vector<float>& SV_z,
                const std::vector<float>& Hadron_GVx,
                const std::vector<float>& Hadron_GVy,
                const std::vector<float>& Hadron_GVz) {
    // computeDistanceMatrix
    // Returns
    // distances = Matrix of Euclidean distances between SV and GV
    
    size_t nSV = SV_x.size();
    size_t nHadron = Hadron_GVx.size();
    
    // 2D vector initialized to 999 nSV x nHadron
    std::vector<std::vector<float>> distances(nSV, std::vector<float>(nHadron, 999.0));

    for (size_t i = 0; i < nSV; ++i) {
        for (size_t j = 0; j < nHadron; ++j) {
            float dx = SV_x[i] - Hadron_GVx[j];
            float dy = SV_y[i] - Hadron_GVy[j];
            float dz = SV_z[i] - Hadron_GVz[j];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            distances[i][j] = dist;
        }
    }

    return distances;
}

// Constructor
TrainAnalyzer::TrainAnalyzer(const edm::ParameterSet& iConfig, const ONNXRuntime *cache):
    // Member Initialization
    //esConsumes -> token to EventSetup module
    //consumes -> token to Event data module
	theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
	training_ (iConfig.getUntrackedParameter<bool>("training")),
	TrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
	PVCollT_ (consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("primaryVertices"))),
	SVCollT_ (consumes<edm::View<reco::VertexCompositePtrCandidate>>(iConfig.getUntrackedParameter<edm::InputTag>("secVertices"))),
  	LostTrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("losttracks"))),
	jet_collT_ (consumes<edm::View<reco::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
        beamspotToken_ (consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamspot"))),
	prunedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("pruned"))),
  	packedGenToken_(consumes<edm::View<pat::PackedGenParticle> >(iConfig.getParameter<edm::InputTag>("packed"))),
	mergedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("merged"))),
	TrackPtCut_(iConfig.getUntrackedParameter<double>("TrackPtCut")),
	vtxconfig_(iConfig.getUntrackedParameter<edm::ParameterSet>("vertexfitter")),
    	vtxmaker_(vtxconfig_),
	PupInfoT_ (consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("addPileupInfo"))),
	vtxweight_(iConfig.getUntrackedParameter<double>("vtxweight")),
	clusterizer(new TracksClusteringFromDisplacedSeed(iConfig.getParameter<edm::ParameterSet>("clusterizer")))
	//genmatch_csv_(iConfig.getParameter<edm::FileInPath>("genmatch_csv").fullPath())
{
	edm::Service<TFileService> fs;	
   	tree = fs->make<TTree>("tree", "tree");
}

// Destructor
TrainAnalyzer::~TrainAnalyzer() {}


// Tell ONNXRunTime where is file location
std::unique_ptr<ONNXRuntime> TrainAnalyzer::initializeGlobalCache(const edm::ParameterSet &iConfig) 
{
    return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void TrainAnalyzer::globalEndJob(const ONNXRuntime *cache) {}


int TrainAnalyzer::checkPDG(int abs_pdg)
// Returns:
// 1 B hadron
// 2 D hadron
// 3 S hadron
// 4 Tau
// 5 anything else
{
    static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541,        // Bottom mesons
        5122,                      // Lambda_b0
        5132, 5232, 5332,          // Xi_b-, Xi_b0, Omega_b-
        5142, 5242, 5342,          // Xi_bc0, Xi_bc+, Omega_bc0
        5512, 5532, 5542, 5554     // Xi_bb-, Omega_bb-, Omega_bbc0, Omega_bbb-
    };

    static const std::unordered_set<int> pdgSet_D = {
        411, 421, 431,             // Charmed mesons
        4122,                      // Lambda_c
        4232, 4132, 4332,          // Xi_c+, Xi’c0, Omega_c0
        4412, 4422, 4432, 4444     // Xi_cc+, Xi_cc++, Omega_cc+, Omega_ccc++
    };

    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212,          // Lambda, Sigma+, Sigma0
        3312, 3322, 3334           // Xi-, Xi0, Omega-
    };

    static const std::unordered_set<int> pdgSet_Tau = {15};

    if (pdgSet_B.count(abs_pdg))      return 1;
    if (pdgSet_D.count(abs_pdg))      return 2;
    if (pdgSet_S.count(abs_pdg))      return 3;
    if (pdgSet_Tau.count(abs_pdg))    return 4;
    return 5;
}

int TrainAnalyzer::getDaughterLabel(const reco::GenParticle* dau)
{
    // Label meaning:
    // 0 = Primary (hard scatter)
    // 1 = Pileup
    // 2 = fromB
    // 3 = fromBC
    // 4 = fromC
    // 5 = OtherSecondary (S, τ, conversions)

    static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541, 5122, 5132, 5232, 5332,
        5142, 5242, 5342, 5512, 5532, 5542, 5554
    };
    static const std::unordered_set<int> pdgSet_C = {
        411, 421, 431, 4122, 4232, 4132, 4332,
        4412, 4422, 4432, 4444
    };
    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212, 3312, 3322, 3334
    };
    static const std::unordered_set<int> pdgSet_Tau = {15};

    // --- 1. Collision ID: cleanest PU separation ---
    if (dau->collisionId() > 0)
        return 1; // pileup (non-primary collision)

    // --- 2. Initialize ancestry flags ---
    bool foundB   = false;
    bool foundC   = false;

    // --- 3. Traverse ancestry chain ---
    const reco::Candidate* mom = dau->mother(0);
    int guard = 0;
    while (mom && guard++ < 100) {
        const int apdg = std::abs(mom->pdgId());

        if (pdgSet_B.count(apdg))   foundB   = true;
        if (pdgSet_C.count(apdg))   foundC   = true;

        // stop once we've reached a B hadron (no need to go higher)
        if (foundB) break;

        mom = mom->mother(0);
    }

    // --- 4. Heavy-flavor classification ---
    if (foundB && foundC) return 3; // fromBC
    if (foundB)           return 2; // fromB
    if (foundC)           return 4; // fromC

    // --- 5. Other displaced / secondary sources ---
    const reco::Candidate* mother = dau->mother(0);
    if (mother) {
        int mabs = std::abs(mother->pdgId());
        if (pdgSet_S.count(mabs) || pdgSet_Tau.count(mabs) || mabs == 22)
            return 5;  // strange hadron, τ, or photon conversion
    }

    // --- 6. Otherwise: primary hard scatter ---
    return 0;
}



bool TrainAnalyzer::isGoodVtx(TransientVertex& tVTX){

   reco::Vertex tmpvtx(tVTX);
   //return (tVTX.isValid() &&
   // !tmpvtx.isFake() &&
   // (tmpvtx.nTracks(vtxweight_)>1) &&
   // (tmpvtx.normalizedChi2()>0) &&
   // (tmpvtx.normalizedChi2()<10));
   return tVTX.isValid();
}



std::vector<TransientVertex> TrainAnalyzer::TrackVertexRefit(std::vector<reco::TransientTrack> &Tracks,
                                                            std::vector<TransientVertex> &VTXs)
    //TrackVertexRefit
    // INPUTS
    // - tracks
    // - inputs
    // OUTPUTS
    // - vertex (new collection based on tracks and inputs used as inputs)
{
    AdaptiveVertexFitter theAVF(GeometricAnnealing(3, 256, 0.25),
				                DefaultLinearizationPointFinder(),
                                KalmanVertexUpdator<5>(),
                                KalmanVertexTrackCompatibilityEstimator<5>(),
                                KalmanVertexSmoother());
 
  std::vector<TransientVertex> newVTXs;

  for(std::vector<TransientVertex>::const_iterator sv = VTXs.begin(); sv != VTXs.end(); ++sv){
      GlobalPoint ssv = sv->position();
      reco::Vertex tmpvtx = reco::Vertex(*sv);
      std::vector<reco::TransientTrack> selTrks; 
      for(std::vector<reco::TransientTrack>::const_iterator trk = Tracks.begin(); trk!=Tracks.end(); ++trk){
	  if (trk->track().pt() <= 0 || !trk->track().charge()) continue;
          Measurement1D ip3d = IPTools::absoluteImpactParameter3D(*trk, tmpvtx).second;
          if(ip3d.significance()<5.0 || sv->trackWeight(*trk)>0.5){
              selTrks.push_back(*trk);
          }
      }
      
      if(selTrks.size()>=2){
          TransientVertex newsv = theAVF.vertex(selTrks, ssv);
          if(isGoodVtx(newsv))  newVTXs.push_back(newsv);
      }
  
  }
  return newVTXs;
}

void TrainAnalyzer::vertexMerge(std::vector<TransientVertex>& VTXs, double maxFraction, double minSignificance) 
//merges (removes) close, overlapping vertices based on shared tracks and distance significance.
// INPUTS
// - VTXs : vertex collection
// - maxFraction: max allowed fraction of shared tracks before merging
// - minSignificance: max allowed significance of vertex distance before merging
{
	
    if (VTXs.empty()) return;
    VertexDistance3D dist;
	
   for (auto sv = VTXs.begin(); sv != VTXs.end(); /* no ++ here */) {
	if (!sv->isValid()) {
        sv = VTXs.erase(sv);  // Remove invalid vertex right away
        continue;
        }
        bool shared = false;
        VertexState s1 = sv->vertexState();
        const auto& tracks_i = sv->originalTracks();

        for (auto sv2 = VTXs.begin(); sv2 != VTXs.end(); ++sv2) {
            if (sv == sv2 || !sv2->isValid()) continue;

            VertexState s2 = sv2->vertexState();
            const auto& tracks_j = sv2->originalTracks();

            // Count shared tracks by comparing pt values
            int sharedTracks = 0;
            for (const auto& ti : tracks_i) {
                for (const auto& tj : tracks_j) {
                    double dpt = std::abs(ti.track().pt() - tj.track().pt());
                    if (dpt < 1e-3) {
                        ++sharedTracks;
                        break;  // each ti only counts once
                    }
                }
            }

            double sharedFrac = static_cast<double>(sharedTracks) / std::min(tracks_i.size(), tracks_j.size());
            double sig = dist.distance(s1, s2).significance();

            if (sharedFrac > maxFraction && sig < minSignificance) {
                shared = true;
                break;
            }
        }

        if (shared) {
            sv = VTXs.erase(sv);  // erase returns next valid iterator
        } else {
            ++sv;
        }
    } 
}

std::vector<TransientVertex>
TrainAnalyzer::TrackVertexArbitrator(const reco::Vertex& pv,
                                    const edm::Handle<reco::BeamSpot>& bsH,
                                    const std::vector<TransientVertex>& seedSVs,
                                    std::vector<reco::TransientTrack>& allTTs,
                                    // arbitration cuts
                                    double dRCut,
                                    double distCut,
                                    double sigCut,
                                    double dLenFraction,
                                    double fitterSigmacut,
                                    double fitterTini,
                                    double fitterRatio,
                                    double maxTimeSig,
                                    // track quality cuts
                                    int trackMinLayers,
                                    double trackMinPt,
                                    int trackMinPixels) {
  std::vector<TransientVertex> out;
  VertexDistance3D vdist;
  GlobalPoint ppv(pv.position().x(), pv.position().y(), pv.position().z());

  AdaptiveVertexFitter avf(
      GeometricAnnealing(fitterSigmacut, fitterTini, fitterRatio),
      DefaultLinearizationPointFinder(),
      KalmanVertexUpdator<5>(),
      KalmanVertexTrackCompatibilityEstimator<5>(),
      KalmanVertexSmoother());

  // handle signed dR cut like CMSSW
  double sign = 1.0;
  if (dRCut < 0) {
    sign = -1.0;
  }
  double dR2cut = dRCut * dRCut;

  std::unordered_map<size_t, Measurement1D> cachedIP;

  for (auto const& sv : seedSVs) {

   if (!sv.isValid()) continue;

    // time info from SV
    const double svTime = sv.time();
    const double svTimeCov = sv.positionError().czz();  // cov(3,3)
    const bool svHasTime = (svTimeCov > 0.);

    GlobalPoint ssv(sv.position().x(), sv.position().y(), sv.position().z());
    GlobalVector flightDir = ssv - ppv;

    Measurement1D dlen = vdist.distance(
        pv,
        VertexState(GlobalPoint(sv.position().x(),
                                sv.position().y(),
                                sv.position().z()),
                    sv.positionError()));

    const auto& svTracks = sv.originalTracks();
    std::vector<reco::TransientTrack> selTracks;
    selTracks.reserve(svTracks.size());

    for (size_t i = 0; i < allTTs.size(); ++i) {
      reco::TransientTrack& tt = allTTs[i];
      if (!tt.isValid()) continue;

      // === Track quality cuts (like CMSSW::trackFilterArbitrator) ===
      if (tt.track().hitPattern().trackerLayersWithMeasurement() < trackMinLayers) continue;
      if (tt.track().pt() < trackMinPt) continue;
      if (tt.track().hitPattern().numberOfValidPixelHits() < trackMinPixels) continue;

      tt.setBeamSpot(*bsH);

      // === Track membership weight (use CMSSW-style) ===
      //float weight = 0.;
      //for (const auto& t0 : svTracks) {
      //  double dpt  = std::abs(tt.track().pt()  - t0.track().pt());
      //  double deta = std::abs(tt.track().eta() - t0.track().eta());
      //  double dphi = std::abs(reco::deltaPhi(tt.track().phi(), t0.track().phi()));
      //  if (dpt < 1e-3 && deta < 1e-3 && dphi < 1e-3) {
      //    weight = 1.0;
      //    break;
      //  }
      //}
      float weight = sv.trackWeight(tt);

      // === IP significance wrt PV (cached) ===
      Measurement1D ipv;
      if (cachedIP.count(i)) {
        ipv = cachedIP[i];
      } else {
        auto ipvp = IPTools::absoluteImpactParameter3D(tt, pv);
        cachedIP[i] = ipvp.second;
        ipv = ipvp.second;
      }

      // === Extrapolate track to SV position ===
      AnalyticalImpactPointExtrapolator extrap(tt.field());
      TrajectoryStateOnSurface tsos =
          extrap.extrapolate(tt.impactPointState(), ssv);

      if (!tsos.isValid()) continue;

      GlobalPoint refPoint = tsos.globalPosition();
      GlobalError refErr = tsos.cartesianError().position();

      Measurement1D isv = vdist.distance(
          VertexState(ssv, sv.positionError()),
          VertexState(refPoint, refErr));

      // === ΔR with signed flight direction ===
      float dR2 = Geom::deltaR2(((sign > 0) ? flightDir : -flightDir), tt.track());

      // === Time significance ===
      double timeSig = 0.;
      if (svHasTime && edm::isFinite(tt.timeExt())) {
        double tErr = std::sqrt(std::pow(tt.dtErrorExt(), 2) + svTimeCov);
        timeSig = std::abs(tt.timeExt() - svTime) / tErr;
      }

      // === Arbitration decision ===
      if (weight > 0. ||
          (isv.significance() < sigCut &&
           isv.value() < distCut &&
           isv.value() < dlen.value() * dLenFraction &&
           timeSig < maxTimeSig)) {

        if ((isv.value() < ipv.value()) &&
            isv.value() < distCut &&
            isv.value() < dlen.value() * dLenFraction &&
            dR2 < dR2cut &&
            timeSig < maxTimeSig) {
          selTracks.push_back(tt);
        } else {
          // Extra acceptance for member tracks, like CMSSW
          if (weight > 0.5 &&
              isv.value() <= ipv.value() &&
              dR2 < dR2cut &&
              timeSig < maxTimeSig) {
            selTracks.push_back(tt);
          }
        }
      }
    }

    // === Vertex refit ===
    if (selTracks.size() >= 2) {
      TransientVertex tv = avf.vertex(selTracks, ssv);
      if (tv.isValid()) {
        // Update vertex time like CMSSW if PV covariance is valid
        if (pv.covariance(3, 3) > 0.) {
          svhelper::updateVertexTime(tv);
        }
        out.push_back(std::move(tv));
      }
    }
  }

  return out;
}

inline float TrainAnalyzer::sigmoid(float x) {
    if (std::isnan(x)) return 0.0f;  // handle NaN safely
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}






// ------------ method called for each event  ------------
void TrainAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace pat;

  run_ = iEvent.id().run();
  lumi_ = iEvent.luminosityBlock();
  evt_ = iEvent.id().event();

  const SigMatchEntry* genmatch = nullptr;

  //auto key = std::make_tuple(run_, lumi_, evt_);
  //auto it = sigMatchMap_.find(key);
  //if (it != sigMatchMap_.end()) {
  //    genmatch = &it->second;
  //}

  std::unordered_map<int, std::pair<int,int>> truthByTrack; // trkIdx -> (label, hadidx)
  truthByTrack.reserve(256);

  std::unordered_set<int> matchedTruthTracks;
  matchedTruthTracks.reserve(256);
  
  //if (genmatch) {
  //  for (size_t k = 0; k < genmatch->indices.size(); ++k) {
  //    int trk = genmatch->indices[k];
  //    int lab = genmatch->labels[k];
  //    int hid = genmatch->hadidx[k];
  //    truthByTrack.emplace(trk, std::make_pair(lab, hid));
  //  }
  //}

  nPU = 0;
    //vectors defined in .h
  Hadron_pt.clear();
  Hadron_SVIdx.clear();
  Hadron_SVDistance.clear();
  Hadron_eta.clear();
  Hadron_phi.clear();
  Hadron_GVx.clear();
  Hadron_GVy.clear();
  Hadron_GVz.clear(); 
  nHadrons.clear();
  nGV.clear();
  nGV_Tau.clear();
  nGV_B.clear();
  nGV_S.clear();
  nGV_D.clear();
  GV_flag.clear();
  nDaughters.clear();
  nDaughters_B.clear();
  nDaughters_D.clear();
  Daughters_hadidx.clear();
  Daughters_flav.clear();
  Daughters_label.clear();
  Daughters_pt.clear();
  //Daughters_pdg.clear();
  Daughters_eta.clear();
  Daughters_phi.clear();
  Daughters_charge.clear();

  ntrks.clear();
  trk_ip2d.clear();
  trk_ip3d.clear();
  trk_ipz.clear();
  trk_ipzsig.clear();
  trk_ip2dsig.clear();
  trk_ip3dsig.clear();
  trk_p.clear();
  trk_pt.clear();
  trk_eta.clear();
  trk_phi.clear();
  trk_charge.clear();
  trk_nValid.clear();
  trk_nValidPixel.clear();
  trk_nValidStrip.clear();

  njets.clear();
  jet_pt.clear();
  jet_eta.clear();
  jet_phi.clear();

  trk_1.clear();
  trk_2.clear();
  deltaR.clear();
  dca.clear();
  dca_sig.clear();
  cptopv.clear();
  pvtoPCA_1.clear();
  pvtoPCA_2.clear();
  dotprod_1.clear();
  dotprod_2.clear();
  pair_mom.clear();
  pair_invmass.clear();

  edge_label.clear();
  truth_has_sv.clear();
  truth_has_b.clear();
  truth_has_btoc.clear();
  truth_has_c.clear();

  nSVs.clear();
  SV_x.clear();
  SV_y.clear();
  SV_z.clear();
  SV_pt.clear();
  SV_mass.clear();
  SV_ntrks.clear();
  SVtrk_pt.clear();
  SVtrk_SVIdx.clear();
  SVtrk_eta.clear();
  SVtrk_phi.clear();
  SVtrk_ipz.clear();
  SVtrk_ipzsig.clear();
  SVtrk_ipxy.clear();
  SVtrk_ipxysig.clear();
  SVtrk_ip3d.clear();
  SVtrk_ip3dsig.clear();

  preds.clear();
  cut.clear();

  nSVs_reco.clear();
  SV_x_reco.clear();
  SVrecoTrk_SVrecoIdx.clear();
  SVrecoTrk_pt.clear();
  SVrecoTrk_eta.clear();
  SVrecoTrk_phi.clear();
  SVrecoTrk_ip3d.clear();
  SVrecoTrk_ip2d.clear();
  SV_y_reco.clear();
  SV_z_reco.clear();
  SV_reco_nTracks.clear();
  SV_chi2_reco.clear();
  Hadron_SVRecoIdx.clear();
  Hadron_SVRecoDistance.clear();

  sv_score_sig.clear();
  sv_score_bkg.clear();
  edge_score_sig.clear();
  edge_score_bkg.clear();

  nHad_B.clear();
  nHad_BtoC.clear();
  nHad_C.clear();
  eff_B.clear();
  eff_BtoC.clear();
  eff_C.clear();
  fake_B.clear();
  fake_BtoC.clear();
  fake_C.clear();


  Handle<PackedCandidateCollection> patcan;
  Handle<PackedCandidateCollection> losttracks;
  Handle<edm::View<reco::Jet> > jet_coll;
  Handle<edm::View<reco::GenParticle> > pruned;
  Handle<reco::BeamSpot> beamSpotHandle;
  Handle<edm::View<pat::PackedGenParticle> > packed;
  Handle<edm::View<reco::GenParticle> > merged;
  Handle<reco::VertexCollection> pvHandle;
  Handle<edm::View<reco::VertexCompositePtrCandidate>> svHandle ;
  Handle<std::vector< PileupSummaryInfo > > PupInfo;

  std::vector<reco::Track> alltracks;
  std::vector<reco::Track> alltracks_ForArbitration;

  iEvent.getByToken(TrackCollT_, patcan);
  iEvent.getByToken(LostTrackCollT_, losttracks);
  iEvent.getByToken(jet_collT_, jet_coll);
  iEvent.getByToken(prunedGenToken_,pruned);
  iEvent.getByToken(packedGenToken_,packed);
  iEvent.getByToken(mergedGenToken_, merged);
  iEvent.getByToken(PVCollT_, pvHandle);
  iEvent.getByToken(SVCollT_, svHandle);
  iEvent.getByToken(beamspotToken_, beamSpotHandle);

  //std::cout<<"Merged size:"<<merged->size()<<std::endl;
  //std::cout<<"Packed size:"<<packed->size()<<std::endl;
  //std::cout<<"Pruned size:"<<pruned->size()<<std::endl;

  const auto& theB = &iSetup.getData(theTTBToken);
  reco::Vertex pv = (*pvHandle)[0];



  GlobalVector direction(1,0,0);
  direction = direction.unit();

  iEvent.getByToken(PupInfoT_, PupInfo);
      std::vector<PileupSummaryInfo>::const_iterator PVI;
       for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI){

           int BX = PVI->getBunchCrossing();
           if(BX ==0) {
               nPU = PVI->getTrueNumInteractions();
               continue;
           }

   }


  // From PFCandidates to tracks
  //Fil two arrays of tracks
  // alltracks
  // alltracjs for arbitration
  for (auto const& itrack : *patcan){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> 0.8 && tmptrk.charge()!=0){
            alltracks_ForArbitration.push_back(tmptrk);
           }
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
	       alltracks.push_back(tmptrk);
           }
       }
   }

   for (auto const& itrack : *losttracks){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
        if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> 0.8 && tmptrk.charge()!=0){
            alltracks_ForArbitration.push_back(tmptrk);
           }
        if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
               alltracks.push_back(itrack.pseudoTrack());
           }
       }
   }

    // Store one event-level pt/eta entry for the final dataset config.
   // If no slimmed jet is available, keep the historical preprocess fallback.
   int njet = 0;
   if (jet_coll.isValid()) {
       for (auto const& ijet: *jet_coll) {
           if (njet == 0) {
               jet_pt.push_back(ijet.pt());
               jet_eta.push_back(ijet.eta());
               jet_phi.push_back(ijet.phi());
           }
           njet++;
       }
   }
   if (jet_pt.empty()) {
       jet_pt.push_back(50.0f);
       jet_eta.push_back(0.0f);
       jet_phi.push_back(0.0f);
   }
   njets.push_back(njet);
   
   int ntrk = 0;
   size_t num_tracks = alltracks.size();
   std::vector<reco::TransientTrack> t_trks(num_tracks);
   std::vector<Measurement1D> ip2d_vals(num_tracks);
   std::vector<Measurement1D> ip3d_vals(num_tracks);

   std::vector<int> origToNode(num_tracks, -1);
   std::vector<int> nodeToOrig;
   nodeToOrig.reserve(num_tracks);
   int node = 0;

   for (size_t i = 0; i< num_tracks; ++i) {
        t_trks[i] = (*theB).build(alltracks[i]);
        if (!(t_trks[i].isValid())) continue;

	origToNode[i] = node;
 	nodeToOrig.push_back((int)i);	
 
        ip2d_vals[i] = IPTools::signedTransverseImpactParameter(t_trks[i], direction, pv).second;
        ip3d_vals[i] = IPTools::signedImpactParameter3D(t_trks[i], direction, pv).second;

	double ip_z = t_trks[i].track().dz(pv.position());
        double ip_z_sig = ip_z / t_trks[i].track().dzError();

        trk_ipz.push_back(ip_z);
        trk_ipzsig.push_back(ip_z_sig);
        trk_ip2d.push_back(std::abs(ip2d_vals[i].value()));
        trk_ip3d.push_back(std::abs(ip3d_vals[i].value()));
        //trk_ip2dsig.push_back(std::abs(ip2d_vals[i].significance()));
        //trk_ip3dsig.push_back(std::abs(ip3d_vals[i].significance()));
        trk_ip2dsig.push_back((std::isfinite(ip2d_vals[i].value()) && std::isfinite(ip2d_vals[i].error()) && ip2d_vals[i].error() > 0.f) ? std::abs(ip2d_vals[i].value() / ip2d_vals[i].error()) : 0.f);
	trk_ip3dsig.push_back((std::isfinite(ip3d_vals[i].value()) && std::isfinite(ip3d_vals[i].error()) && ip3d_vals[i].error() > 0.f) ? std::abs(ip3d_vals[i].value() / ip3d_vals[i].error()) : 0.f);
        trk_p.push_back(alltracks[i].p());
        trk_pt.push_back(alltracks[i].pt());
        trk_eta.push_back(alltracks[i].eta());
        trk_phi.push_back(alltracks[i].phi());
        trk_charge.push_back(alltracks[i].charge());
        trk_nValid.push_back(alltracks[i].numberOfValidHits());
        trk_nValidPixel.push_back(alltracks[i].hitPattern().numberOfValidPixelHits());
        trk_nValidStrip.push_back(alltracks[i].hitPattern().numberOfValidStripHits());
        ntrk++;

	node++;
    }
    
    size_t estimated_pairs = num_tracks * num_tracks / 2;  // Approximate number of pairs
    trk_1.reserve(estimated_pairs);
    trk_2.reserve(estimated_pairs);
    deltaR.reserve(estimated_pairs);
    dca.reserve(estimated_pairs);
    dca_sig.reserve(estimated_pairs);
    cptopv.reserve(estimated_pairs);
    pvtoPCA_1.reserve(estimated_pairs);
    pvtoPCA_2.reserve(estimated_pairs);
    dotprod_1.reserve(estimated_pairs);
    dotprod_2.reserve(estimated_pairs);
    pair_mom.reserve(estimated_pairs);
    pair_invmass.reserve(estimated_pairs); 
	
   // Parallelize the outer loop
    #pragma omp parallel for
    for (size_t i = 0; i < num_tracks; ++i) {
        if (!t_trks[i].isValid()) continue;
    
        for (size_t j = i + 1; j < num_tracks; ++j) {
            if (!t_trks[j].isValid()) continue;
    
            float eta1 = alltracks[i].eta();
            float phi1 = alltracks[i].phi();
            float eta2 = alltracks[j].eta();
            float phi2 = alltracks[j].phi();
            float delta_r_val = reco::deltaR(eta1, phi1, eta2, phi2);
    
            //if (delta_r_val < 2e-4 or delta_r_val > 1) continue;
	
	    const float PION_MASS = 0.13957018; // GeV (charged pion mass)
        
            // Get 3-momenta (px, py, pz)
            float px1 = alltracks[i].px();
            float py1 = alltracks[i].py();
            float pz1 = alltracks[i].pz();
            
            float px2 = alltracks[j].px();
            float py2 = alltracks[j].py();
            float pz2 = alltracks[j].pz();

            // Calculate energies (E = sqrt(p² + m²))
            float e1 = sqrt(px1*px1 + py1*py1 + pz1*pz1 + PION_MASS*PION_MASS);
            float e2 = sqrt(px2*px2 + py2*py2 + pz2*pz2 + PION_MASS*PION_MASS);

            // Compute invariant mass
            float sum_px = px1 + px2;
            float sum_py = py1 + py2;
            float sum_pz = pz1 + pz2;
            float sum_e  = e1 + e2;
            
            float inv_mass = sqrt(sum_e*sum_e - (sum_px*sum_px + sum_py*sum_py + sum_pz*sum_pz));

	    //if(inv_mass > 20.0) continue;
    
            float dca_val = -1.0;  // Default invalid value
	    float cptopv_val = -1.0;
            float pvToPCAseed_val = -1.0;
            float pvToPCAtrack_val = -1.0;
            float dotprodTrack_val = -999.0;
            float dotprodSeed_val = -999.0;
            float dcaSig_val = -1.0;
            float pairMomentumMag = -1.0;

	    TwoTrackMinimumDistance minDist;
            if (minDist.calculate(t_trks[i].impactPointState(), t_trks[j].impactPointState())) {

            	VertexDistance3D distanceComputer;
            	auto m = distanceComputer.distance(
            	VertexState(minDist.points().second, t_trks[i].impactPointState().cartesianError().position()),
            	VertexState(minDist.points().first, t_trks[j].impactPointState().cartesianError().position()));
            	dca_val = m.value();
	    	if(m.error() > 0){
	    	    dcaSig_val = m.value() / m.error();
	    	}
	
	    	GlobalPoint cp(minDist.crossingPoint());
                GlobalPoint pvp(pv.x(), pv.y(), pv.z());
   	    	 
	    	GlobalPoint seedPCA = minDist.points().second;  // PCA of track i (seed)
	    	GlobalPoint trackPCA = minDist.points().first;  // PCA of track j
   	    	
	    	pvToPCAseed_val = (seedPCA - pvp).mag();      // Distance PV to seed track's PCA
	    	pvToPCAtrack_val = (trackPCA - pvp).mag();    // Distance PV to other track's PCA
            	
            	// Calculate additional variables
            	cptopv_val = (cp - pvp).mag();
            	dotprodTrack_val = (trackPCA - pvp).unit().dot(t_trks[j].impactPointState().globalDirection().unit());
            	dotprodSeed_val = (seedPCA - pvp).unit().dot(t_trks[i].impactPointState().globalDirection().unit());
            	
            	// Pair momentum
            	GlobalVector pairMomentum((Basic3DVector<float>)(t_trks[i].track().momentum() + t_trks[j].track().momentum()));
            	pairMomentumMag = pairMomentum.mag();
	     }

	     //if (dca_val < 1e-8 or dca_val > 1 or dcaSig_val > 100 or cptopv_val < 4e-4 or cptopv_val > 20 or pvToPCAseed_val > 20 or pvToPCAtrack_val > 20) continue;	
	     //if (pairMomentumMag < 0.05 or pairMomentumMag > 100) continue;
    
            #pragma omp critical
            {
                trk_1.push_back(i);
                trk_2.push_back(j);
                deltaR.push_back(delta_r_val);
                dca.push_back(dca_val);
		dca_sig.push_back(dcaSig_val);
		cptopv.push_back(cptopv_val);
		pvtoPCA_2.push_back(pvToPCAtrack_val);
		pvtoPCA_1.push_back(pvToPCAseed_val);
		dotprod_2.push_back(dotprodTrack_val);
		dotprod_1.push_back(dotprodSeed_val);
		pair_mom.push_back(pairMomentumMag);
		pair_invmass.push_back(inv_mass);
            }
        }
    }

   ntrks.push_back(ntrk);

   int nhads = 0;
   int ngv = 0;
   int ngv_tau = 0;
   int ngv_b = 0;
   int ngv_s = 0;
   int ngv_d = 0;
   std::vector<float> temp_Daughters_pt;
   //std::vector<float> temp_Daughters_pdg;
   std::vector<float> temp_Daughters_eta;
   std::vector<float> temp_Daughters_phi;
   std::vector<int> temp_Daughters_charge;
   std::vector<int> temp_Daughters_hadidx;
   std::vector<int> temp_Daughters_flav;
   std::vector<int> temp_Daughters_label;

   static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541,        // Bottom mesons
        5122,                      // Lambda_b0
        5132, 5232, 5332,          // Xi_b-, Xi_b0, Omega_b-
        5142, 5242, 5342,          // Xi_bc0, Xi_bc+, Omega_bc0
        5512, 5532, 5542, 5554     // Xi_bb-, Omega_bb-, Omega_bbc0, Omega_bbb-
    };

    static const std::unordered_set<int> pdgSet_D = {
        411, 421, 431,             // Charmed mesons
        4122,                      // Lambda_c
        4232, 4132, 4332,          // Xi_c+, Xi’c0, Omega_c0
        4412, 4422, 4432, 4444     // Xi_cc+, Xi_cc++, Omega_cc+, Omega_ccc++
    };

    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212,          // Lambda, Sigma+, Sigma0
        3312, 3322, 3334           // Xi-, Xi0, Omega-
    };

    static const std::unordered_set<int> pdgSet_Tau = {15};

   for (size_t i = 0; i < merged->size(); ++i) 
   {//Prune loop
       temp_Daughters_pt.clear();
       temp_Daughters_eta.clear();
       temp_Daughters_phi.clear();
       temp_Daughters_charge.clear();
       temp_Daughters_hadidx.clear();
       temp_Daughters_label.clear();
       temp_Daughters_flav.clear();

       const reco::Candidate* hadron = &(*merged)[i];
       if (!(hadron->pt() > 0.0 && std::abs(hadron->eta()) < 2.5)) continue;

       int hadPDG = checkPDG(std::abs(hadron->pdgId())); // 1=B, 2=D, 3=S, 4=tau, 5=other

       int nStableCharged = 0;
       float vx = 0;
       float vy = 0;
       float vz = 0;

       // Keep track if  stored GV for this hadron
       bool addedGV = false;

       // ----------------------------------------
       // Loop over all potential daughters
       // ----------------------------------------
       for (size_t k = 0; k < merged->size(); ++k) {
           const reco::GenParticle* dau = &(*merged)[k];
           if (dau->status() != 1) continue;
           if (dau->charge() == 0) continue;
           if (dau->pt() < 0.8) continue;
           if (std::abs(dau->eta()) > 2.5) continue;

           // Trace ancestry to see if 'hadron' is an ancestor
           bool isDescendant = false;
           const reco::Candidate* mom = dau->mother(0);
           while (mom) {
               if (mom == hadron) { isDescendant = true; break; }

               int mPDG = std::abs(mom->pdgId());
               if (pdgSet_B.count(mPDG) || pdgSet_D.count(mPDG) ||
                   pdgSet_S.count(mPDG) || pdgSet_Tau.count(mPDG))
                   break;
               mom = mom->mother(0);
           }
           if (!isDescendant) continue;

           // --- Classify and store daughter ---
           int label = getDaughterLabel(dau);
           nStableCharged++;

           temp_Daughters_pt.push_back(dau->pt());
           temp_Daughters_eta.push_back(dau->eta());
           temp_Daughters_phi.push_back(dau->phi());
           temp_Daughters_charge.push_back(dau->charge());
           temp_Daughters_hadidx.push_back(ngv);     // group ID for hadron
           temp_Daughters_flav.push_back(hadPDG);  // hadron flavor (5 if light)
           temp_Daughters_label.push_back(label);  // classification label (0–5)

           // Use daughter vertex as approximate GV
           vx = dau->vx();
           vy = dau->vy();
           vz = dau->vz();
       }

       // Require atleast 2 stable charged daughters
       if (nStableCharged < 2) continue;

       // -----------------------------------------
       // Store GV + hadron info once
       // -----------------------------------------
       if (!addedGV) {
           nhads++;
           ngv++; // increment hadron group ID
           addedGV = true;

           if (hadPDG == 1) ngv_b++;
           if (hadPDG == 2) ngv_d++;
           if (hadPDG == 3) ngv_s++;
           if (hadPDG == 4) ngv_tau++;

           Hadron_pt.push_back(hadron->pt());
           Hadron_eta.push_back(hadron->eta());
           Hadron_phi.push_back(hadron->phi());
           Hadron_GVx.push_back(vx);
           Hadron_GVy.push_back(vy);
           Hadron_GVz.push_back(vz);
           GV_flag.push_back(nhads - 1);
       }

       // Append daughters (only after GV stored)
       Daughters_pt.insert(Daughters_pt.end(), temp_Daughters_pt.begin(), temp_Daughters_pt.end());
       Daughters_eta.insert(Daughters_eta.end(), temp_Daughters_eta.begin(), temp_Daughters_eta.end());
       Daughters_phi.insert(Daughters_phi.end(), temp_Daughters_phi.begin(), temp_Daughters_phi.end());
       Daughters_charge.insert(Daughters_charge.end(), temp_Daughters_charge.begin(), temp_Daughters_charge.end());
       Daughters_hadidx.insert(Daughters_hadidx.end(), temp_Daughters_hadidx.begin(), temp_Daughters_hadidx.end());
       Daughters_flav.insert(Daughters_flav.end(), temp_Daughters_flav.begin(), temp_Daughters_flav.end());
       Daughters_label.insert(Daughters_label.end(), temp_Daughters_label.begin(), temp_Daughters_label.end());

       nDaughters.push_back(nStableCharged);
       nDaughters_B.push_back(hadPDG == 1 ? nStableCharged : 0);
       nDaughters_D.push_back(hadPDG == 2 ? nStableCharged : 0);
    } //prune loop

  
   nHadrons.push_back(nhads);
   nGV.push_back(ngv);
   nGV_B.push_back(ngv_b);
   nGV_S.push_back(ngv_s);
   nGV_D.push_back(ngv_d);
   nGV_Tau.push_back(ngv_tau);


   int nsvs = 0;
   for(const auto &sv: *svHandle){
	nsvs++;
   	SV_x.push_back(sv.vertex().x());
	SV_y.push_back(sv.vertex().y());
	SV_z.push_back(sv.vertex().z());
	SV_pt.push_back(sv.pt());
	SV_mass.push_back(sv.p4().M());
	SV_ntrks.push_back(sv.numberOfSourceCandidatePtrs());

	for(size_t i =0; i < sv.numberOfSourceCandidatePtrs(); ++i){
	    reco::CandidatePtr trackPtr = sv.sourceCandidatePtr(i);

	    SVtrk_pt.push_back(trackPtr->pt());
	    SVtrk_eta.push_back(trackPtr->eta());
	    SVtrk_phi.push_back(trackPtr->phi());
        SVtrk_SVIdx.push_back(nsvs-1);
        const pat::PackedCandidate* mypacked = dynamic_cast<const pat::PackedCandidate*>(trackPtr.get());
        if (mypacked && mypacked->hasTrackDetails()) {
        const reco::Track& trk = mypacked->pseudoTrack();

        // dz and dz significance
        double dz    = trk.dz(pv.position());
        double dzErr = trk.dzError();
        SVtrk_ipz.push_back(dz);
        SVtrk_ipzsig.push_back(dzErr > 0 ? dz / dzErr : 0);

        // dxy and dxy significance
        double dxy    = trk.dxy(pv.position());
        double dxyErr = trk.dxyError();
        SVtrk_ipxy.push_back(dxy);
        SVtrk_ipxysig.push_back(dxyErr > 0 ? dxy / dxyErr : 0);

        // 3D IP and significance
        reco::TransientTrack ttrk = theB->build(trk);
        auto ip3dPair = IPTools::absoluteImpactParameter3D(ttrk, pv);
        if (ip3dPair.first) {
            SVtrk_ip3d.push_back(ip3dPair.second.value());
            SVtrk_ip3dsig.push_back(ip3dPair.second.significance());
        } else {
            SVtrk_ip3d.push_back(-999);
            SVtrk_ip3dsig.push_back(-999);
        }
    } else {
        // No track details — fill dummy values
        SVtrk_ipz.push_back(-999);
        SVtrk_ipzsig.push_back(-999);
        SVtrk_ipxy.push_back(-999);
        SVtrk_ipxysig.push_back(-999);
        SVtrk_ip3d.push_back(-999);
        SVtrk_ip3dsig.push_back(-999);
    }
	}


   }
   nSVs.push_back(nsvs);
   std::cout<<"Vertices from IVF : "<< nsvs<<std::endl;
   

   // Here perform the matching
   //distances[i][j] = 3D Euclidean distance between:
    //SV i: position (SV_x[i], SV_y[i], SV_z[i])
    //Hadron j: position (Hadron_GVx[j], Hadron_GVy[j], Hadron_GVz[j])

    auto distances = computeDistanceMatrix(SV_x, SV_y, SV_z, Hadron_GVx, Hadron_GVy, Hadron_GVz);
    
    //distances[i][j] is the distance bewteen SV_i and GV_j
    //std::cout<<"Distance SV 0 GV 0 is "<<distances[0][0]<<std::endl;
    //printDistanceMatrix(distances);
    
    auto result = matchHadronsToSV(
                                        distances,
                                        SVtrk_pt, SVtrk_eta, SVtrk_phi, SVtrk_SVIdx,
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_hadidx,
                                        Hadron_GVx.size()
                                        );
    Hadron_SVIdx = result.first;
    Hadron_SVDistance = result.second;



    auto distances_reco = computeDistanceMatrix(SV_x_reco, SV_y_reco, SV_z_reco, Hadron_GVx, Hadron_GVy, Hadron_GVz);
    //distances[i][j] is the distance bewteen SV_i and GV_j
    //std::cout<<"Distance SV 0 GV 0 is "<<distances[0][0]<<std::endl;
    //printDistanceMatrix(distances);
    //std::cout<<"Distances are computed"<<std::endl;
    auto result_reco = matchHadronsToSV(
                                        distances_reco,
                                        SVrecoTrk_pt, SVrecoTrk_eta, SVrecoTrk_phi, SVrecoTrk_SVrecoIdx,
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_hadidx,
                                        Hadron_GVx.size()
                                        );
    Hadron_SVRecoIdx = result_reco.first;
    Hadron_SVRecoDistance = result_reco.second;

       // Build the final training labels directly in the CMSSW ntuple.
   // This replaces the genmatch/trkinfo.py pass: every reconstructed track is
   // matched to the closest charged daughter passing the same pt, eta, dR, and
   // pt-ratio requirements, with label 6 / hadidx -1 / flav -1 as the default.
   const size_t nTrainingTracks = trk_pt.size();
   trk_label.assign(nTrainingTracks, 6);
   trk_hadidx.assign(nTrainingTracks, -1);
   trk_flav.assign(nTrainingTracks, -1);
   trk_delr.assign(nTrainingTracks, std::numeric_limits<float>::infinity());
   trk_ptrat.assign(nTrainingTracks, std::numeric_limits<float>::quiet_NaN());

   for (size_t it = 0; it < nTrainingTracks; ++it) {
       if (trk_pt[it] < 0.5f || std::abs(trk_eta[it]) >= 2.5f) continue;

       float bestDeltaR = std::numeric_limits<float>::infinity();
       int bestDaughter = -1;
       float bestPtRatio = std::numeric_limits<float>::quiet_NaN();

       for (size_t id = 0; id < Daughters_pt.size(); ++id) {
           if (Daughters_pt[id] <= 0.0f) continue;

           const float ptRatio = trk_pt[it] / Daughters_pt[id];
           if (ptRatio < 0.8f || ptRatio > 1.2f) continue;

           const float dr = reco::deltaR(trk_eta[it], trk_phi[it], Daughters_eta[id], Daughters_phi[id]);
           if (dr >= 0.02f || dr >= bestDeltaR) continue;

           bestDeltaR = dr;
           bestDaughter = static_cast<int>(id);
           bestPtRatio = ptRatio;
       }

       if (bestDaughter >= 0) {
           trk_label[it] = Daughters_label[bestDaughter];
           trk_hadidx[it] = Daughters_hadidx[bestDaughter];
           trk_flav[it] = Daughters_flav[bestDaughter];
           trk_delr[it] = bestDeltaR;
           trk_ptrat[it] = bestPtRatio;
       }
   }

   edge_label.reserve(trk_1.size());
   for (size_t ie = 0; ie < trk_1.size(); ++ie) {
       const int src = trk_1[ie];
       const int dst = trk_2[ie];
       const bool inRange = src >= 0 && dst >= 0 &&
           src < static_cast<int>(trk_hadidx.size()) &&
           dst < static_cast<int>(trk_hadidx.size());
       const bool sameHadron = inRange && trk_hadidx[src] >= 0 &&
           trk_hadidx[src] == trk_hadidx[dst] &&
           trk_flav[src] == trk_flav[dst];
       edge_label.push_back(sameHadron ? 1.0f : 0.0f);
   }

   // These event-level truth branches are compatibility placeholders for the
   // downstream DatasetConstructor. The real supervision written here is the
   // per-track trk_label/trk_hadidx/trk_flav and per-edge edge_label.
   truth_has_sv.push_back(0);
   truth_has_b.push_back(0);
   truth_has_btoc.push_back(0);
   truth_has_c.push_back(0);




   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void TrainAnalyzer::beginStream(edm::StreamID) {

	if(!training_)
	{
		tree->Branch("run", &run_, "run/i");
   		tree->Branch("lumi", &lumi_, "lumi/i");
   		tree->Branch("evt", &evt_);
		tree->Branch("nPU", &nPU);
		tree->Branch("nHadrons", &nHadrons);                // Hadrons of GenVertices             
		tree->Branch("Hadron_pt", &Hadron_pt);              // Hadrons of GenVertices
    		tree->Branch("Hadron_SVIdx", &Hadron_SVIdx);              // Hadrons of GenVertices
    		tree->Branch("Hadron_SVDistance", &Hadron_SVDistance);
		tree->Branch("Hadron_eta", &Hadron_eta);            // Hadrons of GenVertices 
		tree->Branch("Hadron_phi", &Hadron_phi);            // Hadrons of GenVertices 
		tree->Branch("Hadron_GVx", &Hadron_GVx);            // Hadrons of GenVertices 
		tree->Branch("Hadron_GVy", &Hadron_GVy);            // Hadrons of GenVertices 
		tree->Branch("Hadron_GVz", &Hadron_GVz);            // Hadrons of GenVertices 
		tree->Branch("nGV", &nGV);
		tree->Branch("nGV_B", &nGV_B);
    		tree->Branch("nGV_Tau", &nGV_Tau);
    		tree->Branch("nGV_S", &nGV_S);
		tree->Branch("nGV_D", &nGV_D);
		tree->Branch("GV_flag", &GV_flag);                  // GV_genHadronIdx 
		tree->Branch("nDaughters", &nDaughters);
		tree->Branch("nDaughters_B", &nDaughters_B);
		tree->Branch("nDaughters_D", &nDaughters_D);
		tree->Branch("Daughters_hadidx", &Daughters_hadidx);
    		tree->Branch("Daughters_flav", &Daughters_flav);        // flavor of the mother of the vertex
		tree->Branch("Daughters_label", &Daughters_label);
		tree->Branch("Daughters_pt", &Daughters_pt);        // from 0.8 on
		tree->Branch("Daughters_eta", &Daughters_eta);
		tree->Branch("Daughters_phi", &Daughters_phi);      // ok
		tree->Branch("Daughters_charge", &Daughters_charge); // ok +-1
	}
	
	tree->Branch("nTrks", &ntrks);                      // all the tracks in the event
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
    	tree->Branch("trk_dz", &trk_ipz);
    	tree->Branch("trk_dzsig", &trk_ipzsig);
	tree->Branch("trk_ip2dsig", &trk_ip2dsig);
    	tree->Branch("trk_ip3dsig", &trk_ip3dsig);
	tree->Branch("trk_p", &trk_p);
	tree->Branch("trk_pt", &trk_pt);
	tree->Branch("trk_eta", &trk_eta);
	tree->Branch("trk_phi", &trk_phi);
	tree->Branch("trk_nValid", &trk_nValid);
	tree->Branch("trk_nValidPixel", &trk_nValidPixel);
	tree->Branch("trk_nValidStrip", &trk_nValidStrip);
	tree->Branch("trk_charge", &trk_charge);

        tree->Branch("trk_label", &trk_label);
	tree->Branch("trk_hadidx", &trk_hadidx);
	tree->Branch("trk_flav", &trk_flav);
	tree->Branch("trk_delr", &trk_delr);
	tree->Branch("trk_ptrat", &trk_ptrat);
         
    	tree->Branch("trk_1", &trk_1);                  // needed for edge features
	tree->Branch("trk_2", &trk_2);                  // needed for edge features
	tree->Branch("deltaR", &deltaR);                
	tree->Branch("dca", &dca);
	tree->Branch("dca_sig", &dca_sig);
    	tree->Branch("cptopv", &cptopv);
	tree->Branch("pvtoPCA_1", &pvtoPCA_1);
	tree->Branch("pvtoPCA_2", &pvtoPCA_2);
	tree->Branch("dotprod_1", &dotprod_1);
	tree->Branch("dotprod_2", &dotprod_2);
	tree->Branch("pair_mom", &pair_mom);
	tree->Branch("pair_invmass", &pair_invmass);    // inv mass


        tree->Branch("edge_label", &edge_label);
	tree->Branch("truth_has_sv", &truth_has_sv);
	tree->Branch("truth_has_b", &truth_has_b);
	tree->Branch("truth_has_btoc", &truth_has_btoc);
	tree->Branch("truth_has_c", &truth_has_c);
	tree->Branch("nJets", &njets);
	tree->Branch("jet_pt", &jet_pt);
	tree->Branch("jet_eta", &jet_eta);
	tree->Branch("jet_phi", &jet_phi);

	if(!training_)
	{
		tree->Branch("nSVs", &nSVs);
		tree->Branch("SV_x", &SV_x);
		tree->Branch("SV_y", &SV_y);
		tree->Branch("SV_z", &SV_z);
		tree->Branch("SV_pt", &SV_pt);
		tree->Branch("SV_mass", &SV_mass);
		tree->Branch("SV_ntrks", &SV_ntrks);
		tree->Branch("SVtrk_pt", &SVtrk_pt);
    		tree->Branch("SVtrk_SVIdx", &SVtrk_SVIdx);
		tree->Branch("SVtrk_eta", &SVtrk_eta);
		tree->Branch("SVtrk_phi", &SVtrk_phi);
    		tree->Branch("SVtrk_ipz", &SVtrk_ipz);
    		tree->Branch("SVtrk_ipzsig", &SVtrk_ipzsig);
    		tree->Branch("SVtrk_ipxy", &SVtrk_ipxy);
    		tree->Branch("SVtrk_ipxysig", &SVtrk_ipxysig);
    		tree->Branch("SVtrk_ip3d", &SVtrk_ip3d);
    		tree->Branch("SVtrk_ip3dsig", &SVtrk_ip3dsig);
        	
    		tree->Branch("Edge_probs", &preds);
		tree->Branch("cut_val", &cut);

		tree->Branch("nSVs_reco", &nSVs_reco);
    		tree->Branch("SV_x_reco", &SV_x_reco);
    		tree->Branch("Hadron_SVRecoIdx", &Hadron_SVRecoIdx);              // Hadrons of GenVertices
    		tree->Branch("Hadron_SVRecoDistance", &Hadron_SVRecoDistance);
    		tree->Branch("SVrecoTrk_SVrecoIdx", &SVrecoTrk_SVrecoIdx);
    		tree->Branch("SVrecoTrk_pt", &SVrecoTrk_pt);
    		tree->Branch("SVrecoTrk_eta", &SVrecoTrk_eta);
    		tree->Branch("SVrecoTrk_phi", &SVrecoTrk_phi);
    		tree->Branch("SVrecoTrk_ip3d", &SVrecoTrk_ip3d);
    		tree->Branch("SV_y_reco", &SV_y_reco);
    		tree->Branch("SV_z_reco", &SV_z_reco);
    		tree->Branch("SV_reco_nTracks", &SV_reco_nTracks);
    		tree->Branch("SV_chi2_reco", &SV_chi2_reco);

    		tree->Branch("SV_score_sig", &sv_score_sig);
    		tree->Branch("SV_score_bkg", &sv_score_bkg);
    		tree->Branch("Edge_score_sig", &edge_score_sig);
    		tree->Branch("Edge_score_bkg", &edge_score_bkg);

    		tree->Branch("nHad_B", &nHad_B);
    		tree->Branch("nHad_BtoC", &nHad_BtoC);
    		tree->Branch("nHad_C", &nHad_C);
    		tree->Branch("eff_B", &eff_B);
    		tree->Branch("eff_BtoC", &eff_BtoC);
    		tree->Branch("eff_C", &eff_C);
    		tree->Branch("fake_B", &fake_B);
    		tree->Branch("fake_BtoC", &fake_BtoC);
    		tree->Branch("fake_C", &fake_C);
    	}
    

	//auto strip_chars = [&](std::string& s, const std::string& chars) {
	//  s.erase(std::remove_if(s.begin(), s.end(),
	//                         [&](char c){ return chars.find(c) != std::string::npos; }),
	//          s.end());
	//};
	//
	//auto trim_inplace = [&](std::string& s) {
	//  auto notspace = [](unsigned char c){ return !std::isspace(c); };
	//  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
	//  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
	//};
	//
	//auto parse_int_list = [&](std::string s) -> std::vector<int> {
	//  strip_chars(s, "\"[]");
	//  std::vector<int> out;
	//  std::stringstream ss(s);
	//  std::string tok;
	//  while (std::getline(ss, tok, ',')) {
	//    trim_inplace(tok);
	//    if (!tok.empty()) out.push_back(std::stoi(tok));
	//  }
	//  return out;
	//};

	//auto splitCSVQuoted = [&](const std::string& line,
        //                  std::vector<std::string>& fields,
        //                  size_t expected) -> bool {
	//  fields.clear();
	//  std::string cur;
	//  bool inQuotes = false;
	//
	//  for (size_t i = 0; i < line.size(); ++i) {
	//    char c = line[i];
	//
	//    if (c == '"') {
	//      // handle escaped quote ""
	//      if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
	//        cur.push_back('"');
	//        ++i;
	//      } else {
	//        inQuotes = !inQuotes;
	//      }
	//    } else if (c == ',' && !inQuotes) {
	//      fields.push_back(cur);
	//      cur.clear();
	//    } else {
	//      cur.push_back(c);
	//    }
	//  }
	//  fields.push_back(cur);
	//
	//  return (fields.size() == expected);
	//};
    

	//std::ifstream file(genmatch_csv_);
	//std::string line;
	//unsigned int line_no = 0;
	//std::vector<std::string> fields;
	//
	//while (std::getline(file, line)) {
	//    ++line_no;
	//    if (line.empty()) continue;
	//  
	//    if (!splitCSVQuoted(line, fields, 5)) {
	//      std::cerr << "Line " << line_no
	//                << ": bad CSV field count = " << fields.size()
	//                << "\nLine: " << line << "\n";
	//      continue;
	//    }
	//  
	//    const std::string& run_str    = fields[0];
	//    const std::string& lumi_str   = fields[1];
	//    const std::string& evt_str    = fields[2];
	//    const std::string& labels_str = fields[3];
	//    const std::string& hadidx_str = fields[4];
	//  
	//    if (run_str == "run") continue; // header
	//  
	//    const unsigned int run  = std::stoul(run_str);
	//    const unsigned int lumi = std::stoul(lumi_str);
	//    const unsigned int evt  = std::stoul(evt_str);
	//  
	//    const std::vector<int> trk_labels = parse_int_list(labels_str);
	//    const std::vector<int> trk_hadidx = parse_int_list(hadidx_str);
	//  
	//    if (trk_labels.size() != trk_hadidx.size()) {
	//      std::cerr << "Line " << line_no
	//                << ": size mismatch labels=" << trk_labels.size()
	//                << " hadidx=" << trk_hadidx.size() << "\n";
	//      continue;
	//    }
 
	//
	//    SigMatchEntry entry;
	//    entry.indices.reserve(trk_labels.size());
	//    entry.labels.reserve(trk_labels.size());
	//    entry.hadidx.reserve(trk_labels.size());
	//
	//    for (size_t i = 0; i < trk_labels.size(); ++i) {
	//      const int lab = trk_labels[i];
	//      if (lab == 2 || lab == 3 || lab == 4) {
	//        entry.indices.push_back(static_cast<int>(i));  // track index
	//        entry.labels.push_back(lab);                   // label at i
	//        entry.hadidx.push_back(trk_hadidx[i]);         // hadidx at i
	//      }
	//    }
	//
	//    sigMatchMap_[{run, lumi, evt}] = std::move(entry);
	//
	//  } 
   	
}


// ------------ method called once each job just after ending the event loop  ------------
//void TrainAnalyzer::endJob() {
//}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrainAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc);
  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrainAnalyzer);
