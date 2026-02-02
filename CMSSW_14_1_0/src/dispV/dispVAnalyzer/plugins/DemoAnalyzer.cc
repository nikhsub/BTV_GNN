// -*- C++ -*-
//
// Package:    Demo/DemoAnalyzer
// Class:      DemoAnalyzer
//
/**\class DemoAnalyzer DemoAnalyzer.cc Demo/DemoAnalyzer/plugins/DemoAnalyzer.cc

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

#include "dispV/dispVAnalyzer/interface/DemoAnalyzer.h"
#include "dispV/dispVAnalyzer/interface/matchedHadronsToSV.h"
#include <iostream>
#include <omp.h>
#include <unordered_set>
#include <iomanip>
#include <unordered_map>

// function to get 3D distance bewtween all SV and all GV
std::vector<std::vector<float>> computeDistanceMatrix(
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
DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig, const ONNXRuntime *cache):
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
	TrackPredCut_(iConfig.getUntrackedParameter<double>("TrackPredCut")),
	vtxconfig_(iConfig.getUntrackedParameter<edm::ParameterSet>("vertexfitter")),
    	vtxmaker_(vtxconfig_),
	PupInfoT_ (consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("addPileupInfo"))),
	vtxweight_(iConfig.getUntrackedParameter<double>("vtxweight")),
	clusterizer(new TracksClusteringFromDisplacedSeed(iConfig.getParameter<edm::ParameterSet>("clusterizer"))),
	genmatch_csv_(iConfig.getParameter<edm::FileInPath>("genmatch_csv").fullPath())
{
	edm::Service<TFileService> fs;	
	//usesResource("TFileService");
   	tree = fs->make<TTree>("tree", "tree");
}

// Destructor
DemoAnalyzer::~DemoAnalyzer() {}


// Tell ONNXRunTime where is file location
std::unique_ptr<ONNXRuntime> DemoAnalyzer::initializeGlobalCache(const edm::ParameterSet &iConfig) 
{
    return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DemoAnalyzer::globalEndJob(const ONNXRuntime *cache) {}


int DemoAnalyzer::checkPDG(int abs_pdg)
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

int DemoAnalyzer::getDaughterLabel(const reco::GenParticle* dau)
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



bool DemoAnalyzer::isGoodVtx(TransientVertex& tVTX){

   reco::Vertex tmpvtx(tVTX);
   //return (tVTX.isValid() &&
   // !tmpvtx.isFake() &&
   // (tmpvtx.nTracks(vtxweight_)>1) &&
   // (tmpvtx.normalizedChi2()>0) &&
   // (tmpvtx.normalizedChi2()<10));
   return tVTX.isValid();
}



std::vector<TransientVertex> DemoAnalyzer::TrackVertexRefit(std::vector<reco::TransientTrack> &Tracks,
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

void DemoAnalyzer::vertexMerge(std::vector<TransientVertex>& VTXs, double maxFraction, double minSignificance) 
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
DemoAnalyzer::TrackVertexArbitrator(const reco::Vertex& pv,
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

inline float DemoAnalyzer::sigmoid(float x) {
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
void DemoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace pat;

  run_ = iEvent.id().run();
  lumi_ = iEvent.luminosityBlock();
  evt_ = iEvent.id().event();

  const SigMatchEntry* genmatch = nullptr;

  auto key = std::make_tuple(run_, lumi_, evt_);
  auto it = sigMatchMap_.find(key);
  if (it != sigMatchMap_.end()) {
      genmatch = &it->second;
  }

  std::unordered_map<int, std::pair<int,int>> truthByTrack; // trkIdx -> (label, hadidx)
  truthByTrack.reserve(256);

  std::unordered_set<int> matchedTruthTracks;
  matchedTruthTracks.reserve(256);
  
  if (genmatch) {
    for (size_t k = 0; k < genmatch->indices.size(); ++k) {
      int trk = genmatch->indices[k];
      int lab = genmatch->labels[k];
      int hid = genmatch->hadidx[k];
      truthByTrack.emplace(trk, std::make_pair(lab, hid));
    }
  }

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

  //njets.clear();
  //jet_pt.clear();
  //jet_eta.clear();
  //jet_phi.clear();
  trk_i.clear();
  trk_j.clear();
  deltaR.clear();
  dca.clear();
  dca_sig.clear();
  cptopv.clear();
  pvtoPCA_i.clear();
  pvtoPCA_j.clear();
  dotprod_i.clear();
  dotprod_j.clear();
  pair_mom.clear();
  pair_invmass.clear();

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
    //const reco::BeamSpot& beamSpot = *beamSpotHandle;  
   //int njet = 0;
   //for (auto const& ijet: *jet_coll){
   //	jet_pt.push_back(ijet.pt());
   //     jet_eta.push_back(ijet.eta());
   //     jet_phi.push_back(ijet.phi());
   //     njet++;
   //}
   //njets.push_back(njet);
   
   int ntrk = 0;
   size_t num_tracks = alltracks.size();
   std::vector<reco::TransientTrack> t_trks(num_tracks);
   std::vector<Measurement1D> ip2d_vals(num_tracks);
   std::vector<Measurement1D> ip3d_vals(num_tracks);

   for (size_t i = 0; i< num_tracks; ++i) {
        t_trks[i] = (*theB).build(alltracks[i]);
        if (!(t_trks[i].isValid())) continue;
        ip2d_vals[i] = IPTools::signedTransverseImpactParameter(t_trks[i], direction, pv).second;
        ip3d_vals[i] = IPTools::signedImpactParameter3D(t_trks[i], direction, pv).second;

	double ip_z = t_trks[i].track().dz(pv.position());
        double ip_z_sig = ip_z / t_trks[i].track().dzError();

        trk_ipz.push_back(ip_z);
        trk_ipzsig.push_back(ip_z_sig);
        trk_ip2d.push_back(std::abs(ip2d_vals[i].value()));
        trk_ip3d.push_back(std::abs(ip3d_vals[i].value()));
        trk_ip2dsig.push_back(std::abs(ip2d_vals[i].significance()));
        trk_ip3dsig.push_back(std::abs(ip3d_vals[i].significance()));
        trk_p.push_back(alltracks[i].p());
        trk_pt.push_back(alltracks[i].pt());
        trk_eta.push_back(alltracks[i].eta());
        trk_phi.push_back(alltracks[i].phi());
        trk_charge.push_back(alltracks[i].charge());
        trk_nValid.push_back(alltracks[i].numberOfValidHits());
        trk_nValidPixel.push_back(alltracks[i].hitPattern().numberOfValidPixelHits());
        trk_nValidStrip.push_back(alltracks[i].hitPattern().numberOfValidStripHits());
        ntrk++;
    }
    
    size_t estimated_pairs = num_tracks * num_tracks / 2;  // Approximate number of pairs
    trk_i.reserve(estimated_pairs);
    trk_j.reserve(estimated_pairs);
    deltaR.reserve(estimated_pairs);
    dca.reserve(estimated_pairs);
    dca_sig.reserve(estimated_pairs);
    cptopv.reserve(estimated_pairs);
    pvtoPCA_i.reserve(estimated_pairs);
    pvtoPCA_j.reserve(estimated_pairs);
    dotprod_i.reserve(estimated_pairs);
    dotprod_j.reserve(estimated_pairs);
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
    
            if (delta_r_val > 1) continue;
	
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

	    if(inv_mass > 20.0) continue;
    
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

	     if (dca_val > 1 or dcaSig_val > 100 or cptopv_val > 20 or pvToPCAseed_val > 20 or pvToPCAtrack_val > 20) continue;	
	     if (pairMomentumMag < 0.05 or pairMomentumMag > 100) continue;
    
            #pragma omp critical
            {
                trk_i.push_back(i);
                trk_j.push_back(j);
                deltaR.push_back(delta_r_val);
                dca.push_back(dca_val);
		dca_sig.push_back(dcaSig_val);
		cptopv.push_back(cptopv_val);
		pvtoPCA_j.push_back(pvToPCAtrack_val);
		pvtoPCA_i.push_back(pvToPCAseed_val);
		dotprod_j.push_back(dotprodTrack_val);
		dotprod_i.push_back(dotprodSeed_val);
		pair_mom.push_back(pairMomentumMag);
		pair_invmass.push_back(inv_mass);
            }
        }
    }

   ntrks.push_back(ntrk);

  
   if(!training_)
   {
   	std::vector<std::vector<float>> track_features;
   	std::vector<std::vector<float>> edge_features;
   	std::vector<int64_t> edge_i, edge_j;

   	// Step 1: Build track_features, and track bad ones
   	for (size_t i = 0; i < static_cast<size_t>(ntrk); ++i) {
   	    std::vector<float> features = {
   	        trk_eta[i],
   	        trk_phi[i],
   	        trk_ip2d[i],
   	        trk_ip3d[i],
   	        trk_ipz[i],
   	        trk_ipzsig[i],
   	        trk_ip2dsig[i],
   	        trk_ip3dsig[i],
   	        trk_p[i],
   	        trk_pt[i],
   	        static_cast<float>(trk_nValid[i]),
   	        static_cast<float>(trk_nValidPixel[i]),
   	        static_cast<float>(trk_nValidStrip[i]),
   	        static_cast<float>(trk_charge[i])
   	    };
   	
   	    bool has_nan = false;
   	    for (float val : features) {
   	        if (!std::isfinite(val)) {
   	            has_nan = true;
   	            break;
   	        }
   	    }
   	
   	    if (has_nan) {
   	        features = {
   	         -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f,  // 10 dummy float features
   	         -1.0f, -1.0f, -1.0f,  // 3 dummy int features as float
   	         -3.0f          // charge dummy
   	        };
   	    }
   	
   	    track_features.push_back(features);
   	}

   	 
   	
   	for (size_t idx = 0; idx < trk_i.size(); ++idx) {
   	    int i = trk_i[idx];
   	    int j = trk_j[idx];

   	    edge_i.push_back(i);
   	    edge_j.push_back(j);
   	    
   	     edge_features.push_back({
   	     dca[idx],
   	     deltaR[idx],
   	     dca_sig[idx],
   	     cptopv[idx],
   	     pvtoPCA_i[idx],
   	     pvtoPCA_j[idx],
   	     dotprod_i[idx],
   	     dotprod_j[idx],
   	     pair_mom[idx],
   	     pair_invmass[idx]
   	     }); 
   	 }

   	std::vector<float> x_in_flat;
   	x_in_flat.reserve(track_features.size() * 14);
   	for (const auto& feat : track_features)
   	    x_in_flat.insert(x_in_flat.end(), feat.begin(), feat.end());

   	std::vector<float> edge_index_flat_f;
   	edge_index_flat_f.reserve(edge_i.size() * 2);
   	for (size_t k = 0; k < edge_i.size(); ++k) {
   	    edge_index_flat_f.push_back(static_cast<float>(edge_i[k]));
   	}

   	for (size_t k = 0; k < edge_j.size(); ++k) {
   	    edge_index_flat_f.push_back(static_cast<float>(edge_j[k]));
   	}

   	std::vector<float> edge_attr_flat;
   	edge_attr_flat.reserve(edge_features.size() * 10);
   	for (const auto& feat : edge_features)
   	    edge_attr_flat.insert(edge_attr_flat.end(), feat.begin(), feat.end());

   	 //std::cout << "track_features.size(): " << track_features.size() << "\n";
   	 //std::cout << "x_in_flat.size(): " << x_in_flat.size() << "\n";
   	 //std::cout << "edge_index_flat_f.size(): " << edge_index_flat_f.size() << "\n";
   	 //std::cout << "edge_attr_flat.size(): " << edge_attr_flat.size() << "\n";
   	 //std::cout << "edge_i.size(): " << edge_i.size() << "\n";
   	 //std::cout << "edge_features.size(): " << edge_features.size() << "\n";


   	   
   	// === 4. Set input names and feed data ===
   	std::vector<std::string> input_names_ = {"x_in", "edge_index", "edge_attr"};

   	std::vector<std::vector<int64_t>> input_shapes_ = {
   	 {1, static_cast<int64_t>(track_features.size()), 14},        // x_in
   	 {1, 2, static_cast<int64_t>(edge_i.size())},                // edge_index
   	 {1, static_cast<int64_t>(edge_features.size()), 10}         // edge_attr
   	};
   	   
   	std::vector<std::vector<float>> data_ = {
   	 x_in_flat,
   	 edge_index_flat_f,
   	 edge_attr_flat
   	};


   	std::vector<std::vector<float>> output = globalCache()->run(input_names_, data_, input_shapes_);

   	const std::vector<float>& node_logits_flat = output[0];
   	const std::vector<float>& edge_logits_flat = output[1];
   	const std::vector<float>& node_probs_flat  = output[2]; // [N*7]
   	const std::vector<float>& edge_probs_flat  = output[3]; // [E]

	preds = edge_probs_flat;


// 	  std::cerr << "node_probs_flat size = " << node_probs_flat.size() << "\n";
//s	td::cerr << "edge_probs_flat size = " << edge_probs_flat.size() << "\n";
//
//f	loat minNP=1e9, maxNP=-1e9;
//f	or (float v : node_probs_flat) { minNP = std::min(minNP,v); maxNP = std::max(maxNP,v); }
//s	td::cerr << "node_probs range: [" << minNP << ", " << maxNP << "]\n";
//
//f	loat minEP=1e9, maxEP=-1e9;
//f	or (float v : edge_probs_flat) { minEP = std::min(minEP,v); maxEP = std::max(maxEP,v); }
//s	td::cerr << "edge_probs range: [" << minEP << ", " << maxEP << "]\n";
   	



//B	uild IVF - like
   	std::unordered_set<size_t> selected_indices;
   	for (size_t i = 0; i < alltracks.size(); ++i) {
   	    //const reco::Track& trk = alltracks[i];

   	    //double dz = trk.dz(pv.position());
   	    //if (
   	    //         (alltracks[i].pt() > 0.8) && 
   	    //         (std::abs(alltracks[i].eta())<2.5) //NEED TO REMAP TRACKS AND EDGES IF UNCOMMENTED
   	    //          && (std::abs(dz) < 0.3)
   	    // )
   	    //     {
   	    //     selected_indices.insert(i);

   	    //     }
   	    selected_indices.insert(i);
   	}

   	std::cout << "selected_indices..size(): " << selected_indices.size() << "\n";
   	std::cout << "alltracks.size(): " << alltracks.size() << "\n";
   	
   	//IVF is running on tracks passing threshold on GNN score
   	std::unordered_set<size_t> expanded_indices = selected_indices;  // will include nearby tracks too

   	   //IVF is running on genMatch tracks
   	//std::unordered_set<size_t> expanded_indices(matched_indices.begin(), matched_indices.end());

   	//std::cout << "matched_indices..size(): " << matched_indices.size() << "\n";
   	//
   	std::vector<reco::TransientTrack> t_trks_SV;
   	std::vector<size_t> t_trks_SV_indices;

   	// Step 3: Fill t_trks_SV with the full expanded set
   	t_trks_SV.clear();  // if not already empty
   	t_trks_SV_indices.clear();

   	for (size_t idx : expanded_indices) {
   	    t_trks_SV.push_back(t_trks[idx]);
   	    t_trks_SV_indices.push_back(idx);
   	}

	constexpr int C = 7;
	int N = t_trks_SV.size();
	int E = edge_probs_flat.size();
		
	// ------------------------------
	// Tunable parameters (start here)
	// ------------------------------
	const float NodeHFcut      = 0.30f;  // pHF = p2+p3+p4
	const float NodeMaxSVcut   = 0.10f;  // max(p2,p3,p4)
	const float MarginCut      = 0.075f;  // maxSV - maxBkg
	
	const float EdgeMin        = 0.30f;  // keep edges above this as "present" for connectivity
	const float EdgeStrongCut  = 0.60f;  // "strong" internal support
	const int   MinClusterSize = 3;
	
	// k-core-ish pruning inside each component
	const int   MinStrongNbrs  = 1;      // for small clusters; try 2 if you have larger comps
	const float MinMeanEdge    = 0.40f;  // mean incident edge (within comp) requirement
	const int   MaxPruneIters  = 10;
	
	// ------------------------------
	// Helpers
	// ------------------------------
	auto nodeProb = [&](int i, int c) -> float {
	  return node_probs_flat[i * C + c];
	};
	
	auto pHF = [&](int i) -> float {
	  return nodeProb(i,2) + nodeProb(i,3) + nodeProb(i,4);
	};
	
	auto pSVbest = [&](int i) -> float {
	  return std::max({nodeProb(i,2), nodeProb(i,3), nodeProb(i,4)});
	};
	
	auto pBbest = [&](int i) -> float {
	  // adjust these background classes to match your label scheme
	  return std::max({nodeProb(i,0), nodeProb(i,1),  nodeProb(i,6)});
	};
	
	auto isHF = [&](int i) -> bool {
	  const float hf   = pHF(i);
	  const float svmx = pSVbest(i);
	  const float bmx  = pBbest(i);
	  return (hf > NodeHFcut) && (svmx > NodeMaxSVcut) && ((svmx - bmx) > MarginCut);
	};

	auto nodeArgMax = [&](int i) -> int {
          int bestc = 0;
          float bestp = node_probs_flat[i*C + 0];
          for (int c = 1; c < C; ++c) {
            float p = node_probs_flat[i*C + c];
            if (p > bestp) { bestp = p; bestc = c; }
          }
          return bestc;
        };


	// edge index accessors
	auto edgeSrc = [&](int e) -> int { return static_cast<int>(edge_index_flat_f[e]); };
	auto edgeDst = [&](int e) -> int { return static_cast<int>(edge_index_flat_f[E + e]); };
	
	// pack(u,v) helper for unordered_map key
	auto pack = [&](int a, int b) -> uint64_t {
	  return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
	};
	struct U64Hash { size_t operator()(uint64_t x) const noexcept { return std::hash<uint64_t>{}(x); } };
	
	// ------------------------------
	// (1) Node preselection mask
	// ------------------------------
	std::vector<char> keepNode(N, 0);
	for (int i = 0; i < N; ++i) {
	  keepNode[i] = isHF(i) ? 1 : 0;
	}
	
	// If you want to allow non-HF nodes to remain attachable later, you can loosen this,
	// but for "first apply node cuts", we hard mask here.
	
	// ------------------------------
	// (2) Build fast edge score lookup (only among kept nodes, only edges above EdgeMin)
	// ------------------------------
	std::unordered_map<uint64_t, float, U64Hash> edgeScore;
	edgeScore.reserve((size_t)E * 2);
	
	for (int e = 0; e < E; ++e) {
	  int u = edgeSrc(e);
	  int v = edgeDst(e);
	  if (u < 0 || u >= N || v < 0 || v >= N) continue;
	  if (!keepNode[u] || !keepNode[v]) continue;
	
	  float pe = edge_probs_flat[e];
	  if (std::isnan(pe)) continue;
	  if (pe < EdgeMin) continue;
	
	  edgeScore[pack(u,v)] = pe;
	  edgeScore[pack(v,u)] = pe;
	}
	
	// ------------------------------
	// (3) Build adjacency on kept nodes using "present" edges (pe >= EdgeMin)
	// ------------------------------
	std::vector<std::vector<int>> adj(N);
	for (auto const& kv : edgeScore) {
	  uint64_t key = kv.first;
	  int u = int(key >> 32);
	  int v = int(uint32_t(key));
	  // edgeScore has both directions; we'll push both; ok for DFS visited
	  adj[u].push_back(v);
	}
	
	// Optional: de-duplicate adjacency lists
	for (int u = 0; u < N; ++u) {
	  auto& nb = adj[u];
	  if (nb.empty()) continue;
	  std::sort(nb.begin(), nb.end());
	  nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
	}
	
	// ------------------------------
	// (4) Connected components on the masked graph (kept nodes only)
	// ------------------------------
	std::vector<int> visited(N, 0);
	std::vector<std::vector<int>> comps;
	comps.reserve(N);
	
	std::vector<int> stack;
	stack.reserve(N);
	
	for (int start = 0; start < N; ++start) {
	  if (!keepNode[start]) continue;
	  if (visited[start]) continue;
	  if (adj[start].empty()) continue; // ignore isolated kept nodes (change if you want singletons)
	  visited[start] = 1;
	
	  comps.emplace_back();
	  auto& comp = comps.back();
	
	  stack.clear();
	  stack.push_back(start);
	
	  while (!stack.empty()) {
	    int u = stack.back();
	    stack.pop_back();
	    comp.push_back(u);
	
	    for (int v : adj[u]) {
	      if (!keepNode[v]) continue;
	      if (!visited[v]) {
	        visited[v] = 1;
	        stack.push_back(v);
	      }
	    }
	  }
	}
	
	// ------------------------------
	// (5) Prune each component to be "strongly internally supported"
	//     (k-core-ish using strong edges + mean incident edge, iterated)
	// ------------------------------
	auto pruneComp = [&](std::vector<int>& comp) {
	  if ((int)comp.size() < MinClusterSize) { comp.clear(); return; }
	
	  std::unordered_set<int> in;
	  in.reserve(comp.size() * 2);
	
	  for (int iter = 0; iter < MaxPruneIters; ++iter) {
	    in.clear();
	    for (int u : comp) in.insert(u);
	
	    bool changed = false;
	    std::vector<int> keep;
	    keep.reserve(comp.size());
	
	    for (int u : comp) {
	      int strong = 0;
	      int present = 0;
	      float sum = 0.f;
	
	      for (int v : comp) {
	        if (v == u) continue;
	        auto it = edgeScore.find(pack(u,v));
	        if (it == edgeScore.end()) continue;
	        float pe = it->second;
	        present++;
	        sum += pe;
	        if (pe >= EdgeStrongCut) strong++;
	      }
	
	      float mean = (present > 0) ? (sum / float(present)) : 0.f;
	
	      // Require that the node is not just "connected", but supported by multiple strong neighbors
	      bool ok = true;
	      ok &= (strong >= MinStrongNbrs);
	      ok &= (mean >= MinMeanEdge);
	
	      if (ok) keep.push_back(u);
	      else changed = true;
	    }
	
	    comp.swap(keep);
	    if ((int)comp.size() < MinClusterSize) { comp.clear(); return; }
	    if (!changed) break;
	  }
	};
	
	for (auto& comp : comps) {
	  pruneComp(comp);
	}
	
	// Drop empty / too small
	comps.erase(std::remove_if(comps.begin(), comps.end(),
	                           [&](auto const& c){ return (int)c.size() < MinClusterSize; }),
	           comps.end());


	auto printCompDetails = [&](const std::vector<int>& comp, int compId) {
  std::cout << "===== Component " << compId << " size=" << comp.size() << " =====\n";
  std::cout << "trkIdx  model  truthLabel  truthHadidx\n";
  std::cout << "--------------------------------------\n";

  for (int idx : comp) {
    const int m = nodeArgMax(idx);

    auto jt = truthByTrack.find(idx);
    if (jt != truthByTrack.end()) {
      matchedTruthTracks.insert(idx); 
      const int tLab = jt->second.first;
      const int tHid = jt->second.second;

      std::cout << std::setw(5) << idx << "  "
                << std::setw(5) << m   << "  "
                << std::setw(10) << tLab << "  "
                << std::setw(11) << tHid << "\n";
    } else {
      std::cout << std::setw(5) << idx << "  "
                << std::setw(5) << m   << "  "
                << std::setw(10) << "NA" << "  "
                << std::setw(11) << "NA" << "\n";
    }
  }

  std::cout << "\n";
};

auto printUnmatchedTruth = [&]() {
  std::cout << "\n";
  std::cout << "==================== UNMATCHED GENMATCH TRACKS ====================\n";
  std::cout << "trkIdx  truthLabel  truthHadidx\n";
  std::cout << "--------------------------------------------------------------------\n";

  if (!genmatch) {
    std::cout << "(no genmatch entry for this event)\n\n";
    return;
  }

  int nUnmatched = 0;
  for (size_t k = 0; k < genmatch->indices.size(); ++k) {
    const int trk = genmatch->indices[k];
    if (matchedTruthTracks.find(trk) != matchedTruthTracks.end()) continue;

    ++nUnmatched;
    std::cout << std::setw(5) << trk << "  "
              << std::setw(10) << genmatch->labels[k] << "  "
              << std::setw(11) << genmatch->hadidx[k] << "\n";
  }

  if (nUnmatched == 0) {
    std::cout << "(none)\n";
  }
  std::cout << "====================================================================\n\n";
};


int compId = 0;
for (auto const& comp : comps) {
  printCompDetails(comp, compId++);
}
printUnmatchedTruth();

// 2) Denominator: all unique hadidx per label in the truth
std::array<std::unordered_set<int>, C> totalHadidxByLabel;
for (auto& s : totalHadidxByLabel) s.reserve(256);

if (genmatch) {
  for (size_t k = 0; k < genmatch->indices.size(); ++k) {
    const int lab = genmatch->labels[k];
    const int hid = genmatch->hadidx[k];

    // only labels you care about (2/3/4) and real vertices
    if ((lab == 2 || lab == 3 || lab == 4) && hid >= 0) {
      totalHadidxByLabel[lab].insert(hid);
    }
  }
}

// 3) Numerator: unique hadidx per label that are "matched" by your comps
// match definition: in SOME component, >=2 tracks from same hadidx (and same truth label)
std::array<std::unordered_set<int>, C> matchedHadidxByLabel;
for (auto& s : matchedHadidxByLabel) s.reserve(256);

for (auto const& comp : comps) {
  // per-comp counts of tracks per hadidx for each label
  std::unordered_map<int,int> cnt2, cnt3, cnt4;
  cnt2.reserve(64); cnt3.reserve(64); cnt4.reserve(64);

  for (int idx : comp) {
    // If comp indices are node indices, map them here:
    // int trk = nodeToTrkIdx[idx];
    int trk = idx;

    auto it = truthByTrack.find(trk);
    if (it == truthByTrack.end()) continue;

    const int lab = it->second.first;
    const int hid = it->second.second;
    if (hid < 0) continue;

    const int predLab = nodeArgMax(idx);

    // NEW: require model label matches truth label
    if (predLab != lab) continue;

    if (lab == 2) ++cnt2[hid];
    else if (lab == 3) ++cnt3[hid];
    else if (lab == 4) ++cnt4[hid];
  }

  // mark matched hadidx for this comp if >=2 tracks
  for (auto const& [hid, n] : cnt2) if (n >= 2) matchedHadidxByLabel[2].insert(hid);
  for (auto const& [hid, n] : cnt3) if (n >= 2) matchedHadidxByLabel[3].insert(hid);
  for (auto const& [hid, n] : cnt4) if (n >= 2) matchedHadidxByLabel[4].insert(hid);
}

// 4) Print efficiency per label
auto printEff = [&](int lab) {
  const size_t denom = totalHadidxByLabel[lab].size();
  const size_t num   = matchedHadidxByLabel[lab].size();
  const float eff    = (denom == 0 ? 0.f : float(num) / float(denom));
 
  if(denom > 0){
   std::cout << "label " << lab
            << " hadidxMatched=" << num
            << " hadidxTotal=" << denom
            << " efficiency= " << std::fixed << std::setprecision(2) << eff
            << "\n";
  }
};

std::cout << "====================================================================\n\n";

std::cout << "=== Vertex (hadidx) match efficiency: definition  ===\n";
printEff(2);
printEff(3);
printEff(4);

std::array<int, C> fakeVerticesByLabel{}; // init to 0
std::array<int, C> fakeTracksByLabel{};   // init to 0

for (auto const& comp : comps) {

  // For this component, count unmatched tracks by predicted label
  std::array<int, C> unmatchedCountByPred{}; // per comp

  for (int idx : comp) {
    const int pred = nodeArgMax(idx);

    // If comp indices are node indices, map to original track idx for truth lookup:
    // int trk = nodeToTrkIdx[idx];
    int trk = idx;

    const bool hasTruth = (truthByTrack.find(trk) != truthByTrack.end());
    if (!hasTruth) {
      if (pred >= 0 && pred < C) unmatchedCountByPred[pred]++;
    }
  }

  // A "fake vertex instance" for label L exists in this comp if >=2 unmatched tracks share pred label L
  for (int L = 0; L < C; ++L) {
    if (unmatchedCountByPred[L] >= 2) {
      fakeVerticesByLabel[L] += 1;
      fakeTracksByLabel[L]   += unmatchedCountByPred[L]; // how many unmatched tracks contributed
    }
  }
}

// Print summary
std::cout << "=== Fake vertices (unmatched tracks): definition = >=2 unmatched tracks with same label===\n";
for (int L = 0; L < C; ++L) {
  std::cout << "label " << L
            << " fakeVertexInstances=" << fakeVerticesByLabel[L]
            << " fakeTracksUsed=" << fakeTracksByLabel[L]
            << "\n";
}
	
	
	//Clustering
	//

	std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters;
	clusters.reserve(comps.size());
	std::vector<reco::TransientTrack> t_trks_SV_filtered;
	std::unordered_set<int> used;
	
	
	for (auto const& comp : comps) {
	

	   for (int i : comp) {
   		 used.insert(i);
  	   }
	  // pick seed node = strongest node for that cls
	  int seedIdx = comp[0];
	
	  TracksClusteringFromDisplacedSeed::Cluster aCl;
	  aCl.seedingTrack = t_trks_SV[seedIdx];
	  aCl.seedPoint    = GlobalPoint(pv.x(), pv.y(), pv.z());
	
	  aCl.tracks.clear();
	  aCl.tracks.reserve(comp.size());
	  for (int idx : comp) {
	    aCl.tracks.push_back(t_trks_SV[idx]);
	  }
	
	  clusters.push_back(std::move(aCl));
	}

   	 std::cout << "clusters size" << clusters.size() << std::endl;

	t_trks_SV_filtered.reserve(used.size());

	for (int i : used) {
	  t_trks_SV_filtered.push_back(t_trks_SV[i]);
	}

	
   	 // IVF default clustering

   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters = clusterizer->clusters(pv, t_trks_SV);
   	
   	//MODEL based cluster filtering

   	   
   	//GENMATCHED based clustering
   	
   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clustersAll =
   	// clusterizer->clusters(pv, t_trks_SV);

   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters;
   	////
   	//// helper lambda: match track index by (pt, eta, phi) within tolerance
   	//auto matchIndex = [&](const reco::TransientTrack& trk) -> int {
   	//  const auto& ref = trk.track();
   	//  for (size_t i = 0; i < t_trks_SV.size(); ++i) {
   	//    const auto& cand = t_trks_SV[i].track();
   	//    if (std::abs(cand.pt()  - ref.pt())  < 1e-5 &&
   	//        std::abs(cand.eta() - ref.eta()) < 1e-5 &&
   	//        std::abs(reco::deltaPhi(cand.phi(), ref.phi())) < 1e-5) {
   	//      return static_cast<int>(i);
   	//    }
   	//  }
   	//  return -1;  // not found
   	//};

   	////Keep based on seed track
   	//for (auto& cl : clustersAll) {
   	//  int k = matchIndex(cl.seedingTrack);
   	//  if (k >= 0) {
   	//      size_t origIdx = t_trks_SV_indices[k];
   	//      if (genmatched_indices.count(origIdx)) {
   	//          clusters.push_back(std::move(cl));
   	//      }
   	//  }
   	//}

   	//Keep based on number of tracks that are present in genmatched indices
   	//int nRequiredGenMatchedTracks = 1;  // <-- configurable

   	//for (auto& cl : clustersAll) {
   	//    int nGenMatched = 0;
   	//
   	//    // loop over all tracks in the cluster
   	//    for (const auto& trk : cl.tracks) {
   	//        int k = matchIndex(trk);
   	//        if (k >= 0) {
   	//            size_t origIdx = t_trks_SV_indices[k];
   	//            if (expanded_indices.count(origIdx)) { // check if gen matched
   	//                ++nGenMatched;
   	//                if (nGenMatched >= nRequiredGenMatchedTracks) break;
   	//            }
   	//        }
   	//    }
   	//
   	//    if (nGenMatched >= nRequiredGenMatchedTracks) {
   	//        clusters.push_back(std::move(cl));
   	//    }
   	//}

   	std::vector<TransientVertex> recoVertices;
   	VertexDistanceXY vertTool2D;
   	VertexDistance3D vertTool3D;
   	for (std::vector<TracksClusteringFromDisplacedSeed::Cluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); ++cluster) {
   	       if (cluster->tracks.size()<2) continue; 
   	       
   	       // fit one or more vertices from the tracks in one cluster
   	       std::vector<TransientVertex> tmp_vertices = vtxmaker_.vertices(cluster->tracks);
   	       
   	       
   	       
   	       for (std::vector<TransientVertex>::iterator v = tmp_vertices.begin(); v != tmp_vertices.end(); ++v) {

   	         reco::Vertex tmpvtx(*v);
   	         // Compute flight distance significance w.r.t. primary vertex
   	         // Assume primary vertex is the first vertex in recoVertices
   	         //if (!recoVertices.empty()) {
   	         //    Measurement1D dist2D = vertTool2D.distance(tmpvtx, pv);
   	         //    Measurement1D dist3D = vertTool3D.distance(tmpvtx, pv);

   	         //    // Apply your significance cut
   	         //    if (dist2D.significance() < 2.5 || dist3D.significance() < 0.5) {
   	         //        continue; // skip this vertex
   	         //    }
   	         //}

   	         
   	             recoVertices.push_back(*v);
   	       }
   	 }

	std::cout << "Vertex size after AVR: " << recoVertices.size() << std::endl;


   	 vertexMerge(recoVertices, 0.7, 2);

	std::cout << "Vertex size after merge1: " << recoVertices.size() << std::endl;


   	 double dRCut              = 1000.0;
   	 double distCut            = 1000.0;
   	 double sigCut             = 1000.0; 
   	 double dLenFraction       = 1.0; 
   	 double fitterSigmacut     = 3.0; 
   	 double fitterTini         = 256.0; 
   	 double fitterRatio        = 0.25; 
   	 double maxTimeSig         = 9999.0;
   	 int    trackMinLayers     = 2;
   	 double trackMinPt         = 0.4;
   	 int    trackMinPixels     = 1;
   	     
   	 std::vector<TransientVertex> newVTXs = TrackVertexArbitrator(
   	   pv,
   	   beamSpotHandle,
   	   recoVertices,
   	   t_trks_SV,
   	   dRCut,
   	   distCut,
   	   sigCut,
   	   dLenFraction,
   	   fitterSigmacut,
   	   fitterTini,
   	   fitterRatio,
   	   maxTimeSig,
   	   trackMinLayers,
   	   trackMinPt,
   	   trackMinPixels
   	);
   	//std::vector<TransientVertex> newVTXs = recoVertices;

	std::cout << "Vertex size after arbitration: " << newVTXs.size() << std::endl;

  
   	vertexMerge(newVTXs, 0.2, 10);
   	std::cout << "newVTXs size" << newVTXs.size() << std::endl;


   	int nvtx=0;
   	for(size_t ivtx=0; ivtx<newVTXs.size(); ivtx++){
   		    reco::Vertex tmpvtx(newVTXs[ivtx]);
   	     const auto& vtx = newVTXs[ivtx];
   	     const auto& trks = vtx.originalTracks();
   	     int track_per_sv_counter = 0;
   	     for (const auto& ttrk : trks) {
   	         track_per_sv_counter++;
   	             //const reco::Track& trk = ttrk.track();
   	         //reco::CandidatePtr trackPtr = 
   	         SVrecoTrk_SVrecoIdx.push_back(nvtx);
   	         const reco::Track& trk = ttrk.track();
   	         SVrecoTrk_pt.push_back(trk.pt());
   	         SVrecoTrk_eta.push_back(trk.eta());
   	         SVrecoTrk_phi.push_back(trk.phi());

   	         //reco::TransientTrack ttrk = theB->build(trk);
   	         auto ip3d_trk = IPTools::signedImpactParameter3D(ttrk, direction, pv).second;
   	         auto ip2d_trk = IPTools::signedTransverseImpactParameter(ttrk, direction, pv).second;
   	         SVrecoTrk_ip3d.push_back(ip3d_trk.value());
   	         SVrecoTrk_ip2d.push_back(ip2d_trk.value());




   	     }
   	     nvtx++;
   	     SV_x_reco.push_back(tmpvtx.position().x());
   	     SV_y_reco.push_back(tmpvtx.position().y());
   	     SV_z_reco.push_back(tmpvtx.position().z());
   	     SV_reco_nTracks.push_back(track_per_sv_counter);
   	     SV_chi2_reco.push_back(tmpvtx.normalizedChi2()); 
   	}
   	nSVs_reco.push_back(nvtx);
    }

    
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
       if (!(hadron->pt() > 10.0 && std::abs(hadron->eta()) < 2.5)) continue;

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












   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginStream(edm::StreamID) {

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
         
    	tree->Branch("trk_i", &trk_i);                  // needed for edge features
	tree->Branch("trk_j", &trk_j);                  // needed for edge features
	tree->Branch("deltaR", &deltaR);                
	tree->Branch("dca", &dca);
	tree->Branch("dca_sig", &dca_sig);
    	tree->Branch("cptopv", &cptopv);
	tree->Branch("pvtoPCA_i", &pvtoPCA_i);
	tree->Branch("pvtoPCA_j", &pvtoPCA_j);
	tree->Branch("dotprod_i", &dotprod_i);
	tree->Branch("dotprod_j", &dotprod_j);
	tree->Branch("pair_mom", &pair_mom);
	tree->Branch("pair_invmass", &pair_invmass);    // inv mass


	//tree->Branch("nJets", &njets);
	//tree->Branch("jet_pt", &jet_pt);
    //tree->Branch("jet_eta", &jet_eta);
    //tree->Branch("jet_phi", &jet_phi);

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

    

	auto strip_chars = [&](std::string& s, const std::string& chars) {
	  s.erase(std::remove_if(s.begin(), s.end(),
	                         [&](char c){ return chars.find(c) != std::string::npos; }),
	          s.end());
	};
	
	auto trim_inplace = [&](std::string& s) {
	  auto notspace = [](unsigned char c){ return !std::isspace(c); };
	  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
	  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
	};
	
	auto parse_int_list = [&](std::string s) -> std::vector<int> {
	  strip_chars(s, "\"[]");
	  std::vector<int> out;
	  std::stringstream ss(s);
	  std::string tok;
	  while (std::getline(ss, tok, ',')) {
	    trim_inplace(tok);
	    if (!tok.empty()) out.push_back(std::stoi(tok));
	  }
	  return out;
	};

	auto splitCSVQuoted = [&](const std::string& line,
                          std::vector<std::string>& fields,
                          size_t expected) -> bool {
	  fields.clear();
	  std::string cur;
	  bool inQuotes = false;
	
	  for (size_t i = 0; i < line.size(); ++i) {
	    char c = line[i];
	
	    if (c == '"') {
	      // handle escaped quote ""
	      if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
	        cur.push_back('"');
	        ++i;
	      } else {
	        inQuotes = !inQuotes;
	      }
	    } else if (c == ',' && !inQuotes) {
	      fields.push_back(cur);
	      cur.clear();
	    } else {
	      cur.push_back(c);
	    }
	  }
	  fields.push_back(cur);
	
	  return (fields.size() == expected);
	};
    

	std::ifstream file(genmatch_csv_);
	std::string line;
	unsigned int line_no = 0;
	std::vector<std::string> fields;
	
	while (std::getline(file, line)) {
	    ++line_no;
	    if (line.empty()) continue;
	  
	    if (!splitCSVQuoted(line, fields, 5)) {
	      std::cerr << "Line " << line_no
	                << ": bad CSV field count = " << fields.size()
	                << "\nLine: " << line << "\n";
	      continue;
	    }
	  
	    const std::string& run_str    = fields[0];
	    const std::string& lumi_str   = fields[1];
	    const std::string& evt_str    = fields[2];
	    const std::string& labels_str = fields[3];
	    const std::string& hadidx_str = fields[4];
	  
	    if (run_str == "run") continue; // header
	  
	    const unsigned int run  = std::stoul(run_str);
	    const unsigned int lumi = std::stoul(lumi_str);
	    const unsigned int evt  = std::stoul(evt_str);
	  
	    const std::vector<int> trk_labels = parse_int_list(labels_str);
	    const std::vector<int> trk_hadidx = parse_int_list(hadidx_str);
	  
	    if (trk_labels.size() != trk_hadidx.size()) {
	      std::cerr << "Line " << line_no
	                << ": size mismatch labels=" << trk_labels.size()
	                << " hadidx=" << trk_hadidx.size() << "\n";
	      continue;
	    }
 
	
	    SigMatchEntry entry;
	    entry.indices.reserve(trk_labels.size());
	    entry.labels.reserve(trk_labels.size());
	    entry.hadidx.reserve(trk_labels.size());
	
	    for (size_t i = 0; i < trk_labels.size(); ++i) {
	      const int lab = trk_labels[i];
	      if (lab == 2 || lab == 3 || lab == 4) {
	        entry.indices.push_back(static_cast<int>(i));  // track index
	        entry.labels.push_back(lab);                   // label at i
	        entry.hadidx.push_back(trk_hadidx[i]);         // hadidx at i
	      }
	    }
	
	    sigMatchMap_[{run, lumi, evt}] = std::move(entry);
	
	  } 
   	
}


// ------------ method called once each job just after ending the event loop  ------------
//void DemoAnalyzer::endJob() {
//}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
DEFINE_FWK_MODULE(DemoAnalyzer);

