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
#include <algorithm>
#include <iomanip>




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
	clusterizer(new TracksClusteringFromDisplacedSeed(iConfig.getParameter<edm::ParameterSet>("clusterizer")))
	//genmatch_csv_(iConfig.getParameter<edm::FileInPath>("genmatch_csv").fullPath())
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

/*std::optional<std::tuple<float, float, float>> DemoAnalyzer::isAncestor(const reco::Candidate* ancestor, const reco::Candidate* particle)
//checks if ancestor is an ancestor of particle in the decay chain
// returns the vertex coordinates (x, y, z) of the direct daughter right after the ancestor

{
    // Particle is already the ancestor
    if (ancestor == particle) {
        // Use NaN values to indicate that this is the ancestor but we are not returning its vertex
        return std::make_optional(std::make_tuple(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
    }

    // Otherwise, loop on mothers, if any, and check for the ancestor in the next level up
    for (size_t i = 0; i < particle->numberOfMothers(); i++) {
        auto result = isAncestor(ancestor, particle->mother(i));
        if (result) {
            // If we found a NaN tuple, it means this particle is the child of the ancestor
            if (std::isnan(std::get<0>(*result))) {
                // So, return this particle's vertex since it's the direct descendant
                return std::make_optional(std::make_tuple(particle->vx(), particle->vy(), particle->vz()));
            } else {
                // Otherwise, keep passing up the found vertex coordinates
                return result;
            }
        }
    }

    // If we did not return yet, then particle and ancestor are not relatives
    return std::nullopt;  // Return an empty optional if no ancestor found
}*/



// Check if ancestor is ancestor of particle and give decay point of ancestor
std::optional<std::tuple<float, float, float>> DemoAnalyzer::isAncestor(
    const reco::Candidate* ancestor,
    const reco::Candidate* particle)
{
    const reco::Candidate* current = particle;
    //const reco::Candidate* child = nullptr;

    while (current != nullptr && current->numberOfMothers() > 0) {
        const reco::Candidate* mother = current->mother(0);
        if (mother == ancestor) {
            // Found the ancestor; return the vertex of the current particle (i.e., the direct daughter)
            return std::make_optional(std::make_tuple(current->vx(), current->vy(), current->vz()));
        }
        //child = current;
        current = mother;
    }

    // If we reached here, the ancestor was not found in the chain
    return std::nullopt;
}


int DemoAnalyzer::checkPDG(int abs_pdg)
//Returns 1 for B hadrons
//Returns 2 for D hadrons
//Return 3 for S hadrons
//Returns 0 for anything else
{

	std::vector<int> pdgList_B = { 521, 511, 531, 541, //Bottom mesons
				        5122,   //lambda_b0
                        //5112,   //sigma_b-  // too short  
                        //5212,   //sigma_b0  // too short
                        //5222,   //sigma_b+  // too short
                        5132,   //xi_b-
                        5232,   //xi_b0
                        5332,   //omega_b-
                        5142,   //xi_bc0
                        5242,   //xi_bc+
                        5342,   //omega_bc0
                        5512,   //xi_bb-
                        5532,   //omega_bb-
                        5542,   //Omega_bbc0
                        5554,   //Omega_bbb-
                        };

	std::vector<int> pdgList_D = {411, 421, 431,      // Charmed mesons
                                4122,   //lambda_c  
                                //4222,   //sigma_c++     // too short
                                //4212,   //sigma_c+      // too short
                                //4112,   //sigma_c0      // too short
                                4232,   // csi_c+
                                4132,   // csi'c0
                                4332,   // omega^0_c
                                4412,   //xi_cc^+
                                4422,   //xi_cc^++
                                4432,   //omega^+_cc
                                4444,   //omega^++_ccc
                                };
    std::vector<int> pdgList_Tau = { 
                        15
                        };
    std::vector<int> pdgList_S = {
                                     3122,   //lambda
                                     3222,   //sigma+
                                     3212,   //sigma0
                                     3312,   //csi-
                                     3322,   //csi0
                                     3334,   //omega-
    };
	if(std::find(pdgList_B.begin(), pdgList_B.end(), abs_pdg) != pdgList_B.end()){
		return 1;
	}
	else if(std::find(pdgList_D.begin(), pdgList_D.end(), abs_pdg) != pdgList_D.end()){
	       	return 2;
	}
    else if(std::find(pdgList_S.begin(), pdgList_S.end(), abs_pdg) != pdgList_S.end()){
	       	return 3;
    }
    else if(std::find(pdgList_Tau.begin(), pdgList_Tau.end(), abs_pdg) != pdgList_Tau.end()){
	       	return 4;
	}
	else{
		return 0;
	}
}


//bool DemoAnalyzer::hasDescendantWithId(const reco::Candidate* particle, const std::vector<int>& pdgIds)
// not used
//search for any direct descendant from particle and going below and check if any has a pdgId in the vector list
//{
//    // Base case: If the particle is null, return false
//    if (!particle) {
//        return false;
//    }
//
//    // Loop over all daughters
//    for (size_t i = 0; i < particle->numberOfDaughters(); i++) {
//        const reco::Candidate* daughter = particle->daughter(i);
//        
//        // Check if the current daughter is in the D hadron list
//        if (daughter && std::find(pdgIds.begin(), pdgIds.end(), daughter->pdgId()) != pdgIds.end()) {
//            return true; // Found a D hadron anywhere in the decay chain
//        }
//
//        // Recursively check deeper in the decay chain
//        if (hasDescendantWithId(daughter, pdgIds)) {
//            return true;
//        }
//    }
//
//    return false; // No D hadron found in the decay chain
//}


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

  double dR2cut = dRCut * dRCut;

  for (auto const& sv : seedSVs) {
    if (!sv.isValid()) continue;

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

    std::unordered_map<size_t, Measurement1D> cachedIP;

    for (size_t i = 0; i < allTTs.size(); ++i) {
      reco::TransientTrack& tt = allTTs[i];
      if (!tt.isValid()) continue;

      // === Track quality cuts ===
      if (tt.track().hitPattern().trackerLayersWithMeasurement() < trackMinLayers) continue;
      if (tt.track().pt() < trackMinPt) continue;
      if (tt.track().hitPattern().numberOfValidPixelHits() < trackMinPixels) continue;

      // === Membership check using (pt, eta, phi) match ===
      bool isMember = false;
      for (const auto& t0 : svTracks) {
        double dpt  = std::abs(tt.track().pt()  - t0.track().pt());
        double deta = std::abs(tt.track().eta() - t0.track().eta());
        double dphi = std::abs(reco::deltaPhi(tt.track().phi(), t0.track().phi()));
        if (dpt < 1e-3 && deta < 1e-3 && dphi < 1e-3) {
          isMember = true;
          break;
        }
      }

      tt.setBeamSpot(*bsH);

      Measurement1D ipv;
      if (cachedIP.count(i)) {
        ipv = cachedIP[i];
      } else {
        auto ipvp = IPTools::absoluteImpactParameter3D(tt, pv);
        cachedIP[i] = ipvp.second;
        ipv = ipvp.second;
      }

      AnalyticalImpactPointExtrapolator extrap(tt.field());
      TrajectoryStateOnSurface tsos =
       extrap.extrapolate(tt.impactPointState(),
                          GlobalPoint(sv.position().x(),
                                      sv.position().y(),
                                      sv.position().z()));

      if (!tsos.isValid()) continue;

      GlobalPoint refPoint = tsos.globalPosition();
      GlobalError refErr = tsos.cartesianError().position();

      Measurement1D isv = vdist.distance(
      VertexState(GlobalPoint(sv.position().x(),
                              sv.position().y(),
                              sv.position().z()),
                  sv.positionError()),
      VertexState(refPoint, refErr));

      float dR2 = Geom::deltaR2(flightDir, tt.track());

      double timeSig = 0.;
      if (edm::isFinite(tt.timeExt())) {
        double tErr = std::sqrt(std::pow(tt.dtErrorExt(), 2) + sv.positionError().cxx());
        timeSig = std::abs(tt.timeExt() - sv.time()) / tErr;
      }

      // === Arbitration decision ===
      if (isMember ||
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
        }
      }
    }

    if (selTracks.size() >= 2) {
      TransientVertex tv = avf.vertex(selTracks, ssv);
      if (tv.isValid()) {
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

  //std::vector<int> matched_indices;
	
  //auto key = std::make_tuple(run_, lumi_, evt_);
  //auto it = sigMatchMap_.find(key);
  //if (it != sigMatchMap_.end()) {
  //    matched_indices = it->second;
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
  Daughters_flag.clear();
  Daughters_flav.clear();
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



    //auto ipZPair = IPTools::absoluteImpactParameterZ(t_trks[i], pv);
    double ip_z = t_trks[i].track().dz(pv.position());
    double ip_z_sig = ip_z / t_trks[i].track().dzError();


    trk_ipz.push_back(ip_z);
    trk_ipzsig.push_back(ip_z_sig);
   	trk_ip2d.push_back(ip2d_vals[i].value());
	trk_ip3d.push_back(ip3d_vals[i].value());
	trk_ip2dsig.push_back(ip2d_vals[i].significance());
	trk_ip3dsig.push_back(ip3d_vals[i].significance());
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
    
            if (delta_r_val > 0.4) continue;
	
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

	     if (dca_val > 0.2) continue;	
    
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
            -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f,  // 8 dummy float features
            -1.0f, -1.0f, -1.0f,  // 3 dummy int features as float
            -3.0f          // charge dummy
           };
       }
   
       track_features.push_back(features);
   }

    
   
   for (size_t idx = 0; idx < trk_i.size(); ++idx) {
       int i = trk_i[idx];
       int j = trk_j[idx];

       if (cptopv[idx] >= 100.0 || pair_mom[idx] >= 100.0) continue;

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
   x_in_flat.reserve(track_features.size() * 12);
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
    {1, static_cast<int64_t>(track_features.size()), 12},        // x_in
    {1, 2, static_cast<int64_t>(edge_i.size())},                // edge_index
    {1, static_cast<int64_t>(edge_features.size()), 10}         // edge_attr
   };
      
   std::vector<std::vector<float>> data_ = {
    x_in_flat,
    edge_index_flat_f,
    edge_attr_flat
   };


   std::vector<std::vector<float>> output = globalCache()->run(input_names_, data_, input_shapes_);
   

   // Get logits from output[1]
   std::vector<float> logits_data = output[1];  // output[1] is "node_probs"

   //for (size_t i = 0; i < logits_data.size(); ++i) {
   //    float logit = logits_data[i];
   //    //std::cout << logit << std::endl;
   //    if (std::abs(logit) > 9) {
   //        std::cout << "=== Anomalous Logit ===" << std::endl;
   //        std::cout << "Index " << i << ", Logit: " << logit << std::endl;

   //        const auto& features = track_features[i];
   //        std::cout << "Track Features: [";
   //        for (size_t j = 0; j < features.size(); ++j) {
   //            std::cout << features[j];
   //            if (j != features.size() - 1) std::cout << ", ";
   //        }
   //        std::cout << "]\n";
   //    }
   //}

   std::vector<reco::TransientTrack> t_trks_SV;

   std::vector<std::pair<float, size_t>> score_index_pairs;
   for (size_t i = 0; i < logits_data.size(); ++i) {
       if (std::isnan(logits_data[i])) {
           std::cerr << "Warning: NaN in logit at index " << i;
       }
       float raw_score = sigmoid(logits_data[i]);
       if (std::isnan(raw_score)) {
           std::cerr << "Warning: NaN in sigmoid at index " << i << ", setting to 0.0\n";
       }
       preds.push_back(raw_score);
       score_index_pairs.emplace_back(raw_score, i);  // (score, index)
   }
   //
   //// Step 2: Sort by descending score
   //std::sort(score_index_pairs.begin(), score_index_pairs.end(),
   //          [](const auto& a, const auto& b) { return a.first > b.first; });
   //
   //size_t n_keep = std::max<size_t>(1, score_index_pairs.size() * TrackPredCut_);  // ensure at least one
   //for (size_t k = 0; k < n_keep; ++k) {
   //    size_t idx = score_index_pairs[k].second;
   //    t_trks_SV.push_back(t_trks[idx]);
   //}

   //float event_cut_threshold = score_index_pairs[n_keep - 1].first;
   //cut.push_back(event_cut_threshold);
   //
   float score_threshold = TrackPredCut_; // Tuned based on ROC curve



//Build IVF - like
   std::unordered_set<size_t> selected_indices;
   for (size_t i = 0; i < preds.size(); ++i) {
        const reco::Track& trk = alltracks[i];

        double dz = trk.dz(pv.position());
       if (
                (preds[i] > score_threshold) && 
                (alltracks[i].pt() > 0.8) && 
                (std::abs(alltracks[i].eta())<2.5) && 
                (std::abs(dz) < 0.3)
        )
            {
            selected_indices.insert(i);

            }
   }
   
   // Step 2: Add nearby tracks based on deltaR

   //IVF is running on tracks passing threshold on GNN score
   std::unordered_set<size_t> expanded_indices = selected_indices;  // will include nearby tracks too

   //IVF is running on genMathc tracks
   //std::unordered_set<size_t> expanded_indices(matched_indices.begin(), matched_indices.end());
   
   //for (size_t k = 0; k < deltaR.size(); ++k) {
   //    if (deltaR[k] >= 0.1 and dca[k] >= 0.007 and dca_sig[k] >= 1.0 and cptopv[k] >= 0.9) continue;
   //
   //    size_t i = trk_i[k];
   //    size_t j = trk_j[k];
   //
   //    // If one of them is selected, include the other
   //    if (selected_indices.count(i)) {
   //        expanded_indices.insert(j);
   //    } else if (selected_indices.count(j)) {
   //        expanded_indices.insert(i);
   //    }
   //}
   
   // Step 3: Fill t_trks_SV with the full expanded set
   t_trks_SV.clear();  // if not already empty
   for (size_t idx : expanded_indices) {
       t_trks_SV.push_back(t_trks[idx]);
   }
   
   

    // 1 TracksClusteringFromDisplacedSeed
   std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters = clusterizer->clusters(pv, t_trks_SV);
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
            if (!recoVertices.empty()) {
                Measurement1D dist2D = vertTool2D.distance(tmpvtx, pv);
                Measurement1D dist3D = vertTool3D.distance(tmpvtx, pv);

                // Apply your significance cut
                if (dist2D.significance() < 2.5 || dist3D.significance() < 0.5) {
                    continue; // skip this vertex
                }
            }

            
	        recoVertices.push_back(*v);
          }
    }


    vertexMerge(recoVertices, 0.7, 2);


    double dRCut              = 0.4;
    double distCut            = 0.04;
    double sigCut             = 5.0; 
    double dLenFraction       = 0.333; 
    double fitterSigmacut     = 3.0; 
    double fitterTini         = 256.0; 
    double fitterRatio        = 0.25; 
    double maxTimeSig         = 3.0;
    int    trackMinLayers     = 4;
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


    
   int nhads = 0;
   int ngv = 0;
   int ngv_tau = 0;
   int ngv_b = 0;
   int ngv_s = 0;
   int ngv_d = 0;
   int nd = 0;
   int nd_b = 0;
   int nd_d = 0;
   std::vector<float> temp_Daughters_pt;
   //std::vector<float> temp_Daughters_pdg;
   std::vector<float> temp_Daughters_eta;
   std::vector<float> temp_Daughters_phi;
   std::vector<int> temp_Daughters_charge;
   std::vector<int> temp_Daughters_flag;
   std::vector<int> temp_Daughters_flav;
   

   	std::vector<int> pdgList_B = { 521, 511, 531, 541, //Bottom mesons
				        5122,   //lambda_b0
                        //5112,   //sigma_b-  // too short  
                        //5212,   //sigma_b0  // too short
                        //5222,   //sigma_b+  // too short
                        5132,   //xi_b-
                        5232,   //xi_b0
                        5332,   //omega_b-
                        5142,   //xi_bc0
                        5242,   //xi_bc+
                        5342,   //omega_bc0
                        5512,   //xi_bb-
                        5532,   //omega_bb-
                        5542,   //Omega_bbc0
                        5554,   //Omega_bbb-
                        };
    std::vector<int> pdgList_Tau = { 
                        15
                        };

	std::vector<int> pdgList_D = {411, 421, 431,      // Charmed mesons
                                4122,   //lambda_c  
                                //4222,   //sigma_c++     // too short
                                //4212,   //sigma_c+      // too short
                                //4112,   //sigma_c0      // too short
                                4232,   // csi_c+
                                4132,   // csi'c0
                                4332,   // omega^0_c
                                4412,   //xi_cc^+
                                4422,   //xi_cc^++
                                4432,   //omega^+_cc
                                4444,   //omega^++_ccc
                                };
    std::vector<int> pdgList_S = {
                                     3122,   //lambda
                                     3222,   //sigma+
                                     3212,   //sigma0
                                     3312,   //csi-
                                     3322,   //csi0
                                     3334,   //omega-
    };

   std::unordered_set<int> pdgSet_D(pdgList_D.begin(), pdgList_D.end());
   std::unordered_set<int> pdgSet_B(pdgList_B.begin(), pdgList_B.end());
   std::unordered_set<int> pdgSet_S(pdgList_S.begin(), pdgList_S.end());
   std::unordered_set<int> pdgSet_Tau(pdgList_Tau.begin(), pdgList_Tau.end());


   
   for(size_t i=0; i< merged->size();i++)
   {  //prune loop
   // looping over Hadrons as mothers [i]
    //temp_Daughters_pdg.clear();
	temp_Daughters_pt.clear();
    temp_Daughters_eta.clear();
    temp_Daughters_phi.clear();
    temp_Daughters_charge.clear();
    temp_Daughters_flag.clear();
    temp_Daughters_flav.clear();

	const reco::Candidate* prun_part = &(*merged)[i];
	if(!(prun_part->pt() > 10 && std::abs(prun_part->eta()) < 2.5)) continue;
    // pt and eta requirement are satisfied for the mother
    // pdg of the meson candidate and the mother
	int hadPDG = checkPDG(std::abs(prun_part->pdgId()));

	//int had_parPDG = checkPDG(std::abs(prun_part->mother(0)->pdgId())); GC not used
    
	//if (hadPDG == 1) { // B hadron
    //        int n_charged_nonD_daughters = 0;
	//    for (size_t i = 0; i < prun_part->numberOfDaughters(); ++i) {
    //    	const Candidate* dau = prun_part->daughter(i);
	//	if (!dau) continue; // Safety check
    //            int dau_pdg = std::abs(dau->pdgId());
    //            // pdgSet_D.count(dau_pdg)  returns 1 if dau_pdg in the set, 0 otherwise
    //            if (dau->charge() != 0 && pdgSet_D.count(dau_pdg) == 0) {
    //                n_charged_nonD_daughters++;
    //            }
    //        }
    //    
    //        if (n_charged_nonD_daughters < 2) {
    //            continue; // Skip B hadron — GV will be better captured by D hadron
    //        }
    //    }



    // check if n_stable_charged_daughters >=2
    if (hadPDG >= 1) {  // B or D hadron (or S)
        int n_stable_charged_daughters = 0;


        for (size_t k = 0; k < merged->size(); ++k){

        // k is looping over the genParticles to look for daugh
            const reco::Candidate* current = &(*merged)[k];
            // Apply kinematic + status cuts first
            if (current->status() != 1) continue;
            if (current->charge() == 0) continue;
            if (current->pt() < 0.8) continue;
            if (std::abs(current->eta()) > 2.5) continue;

            bool isDescendant = false;

            while (current->mother(0)) {
                const reco::Candidate* mother = current->mother(0);

                // If we reach prun_part, it's a descendant
                if (mother == prun_part) {
                    isDescendant = true;
                    break;
                }

                // If an intermediate mother is a B or D hadron, stop (displaced decay)
                int mother_pdg = std::abs(mother->pdgId());
                if (pdgSet_B.count(mother_pdg) || pdgSet_D.count(mother_pdg) || pdgSet_S.count(mother_pdg) || pdgSet_Tau.count(mother_pdg)) break;

                current = mother;
            }

            if (isDescendant) {
                n_stable_charged_daughters++;
                if (n_stable_charged_daughters >= 2) break;
            }
        }

    // Not enough valid stable charged daughters → skip this hadron
    if (n_stable_charged_daughters < 2) continue;
    }

	// From here on, only hadrons with 2 stable charged daughters, pt>0.8, eta acceptance, stable, from the same GV
	if(hadPDG > 0)
	{ //if pdg
		nhads++;
		Hadron_pt.push_back(prun_part->pt());
		Hadron_eta.push_back(prun_part->eta());
		Hadron_phi.push_back(prun_part->phi());
		bool addedGV = false;
		int nPack = 0;
		float vx = std::numeric_limits<float>::quiet_NaN();
        float vy = std::numeric_limits<float>::quiet_NaN();
        float vz = std::numeric_limits<float>::quiet_NaN();

		for(size_t j=0; j< merged->size(); j++){
			const Candidate *pack =  &(*merged)[j];
			if(pack==prun_part) continue;
			if(!(pack->status()==1 && pack->pt() > 0.8 && std::abs(pack->eta()) < 2.5 && std::abs(pack->charge()) > 0)) continue;
			//const Candidate * mother = pack->mother(0);
			const Candidate * dau_candidate = pack;
                if(dau_candidate != nullptr){
                    auto GV = isAncestor(prun_part, dau_candidate);  
                    if(GV.has_value()){
                        std::tie(vx, vy, vz) = *GV;
                        if (!std::isnan(vx) && !std::isnan(vy) && !std::isnan(vz)){
                            nPack++;
                            temp_Daughters_pt.push_back(pack->pt());
                            //temp_Daughters_pdg.push_back(pack->pdgId());
                            temp_Daughters_eta.push_back(pack->eta());
                            temp_Daughters_phi.push_back(pack->phi());
                            temp_Daughters_charge.push_back(pack->charge());
                            temp_Daughters_flag.push_back(ngv); //Hadron index
                            temp_Daughters_flav.push_back(hadPDG); //Hadron flav
                
                        }
                
                    }
        
                }
			
		}
	   
		if(nPack >=2){
			if(!addedGV){
				ngv++;
				if(hadPDG==1) ngv_b++;
                if(hadPDG==2) ngv_d++;
                if(hadPDG==3) ngv_s++;
                if(hadPDG==4) ngv_tau++;
			        Hadron_GVx.push_back(vx);
                    Hadron_GVy.push_back(vy); 
                    Hadron_GVz.push_back(vz); 
                    GV_flag.push_back(nhads-1); //Which hadron it belongs to
				addedGV = true;
			}
			Daughters_pt.insert(Daughters_pt.end(), temp_Daughters_pt.begin(), temp_Daughters_pt.end());
            //Daughters_pdg.insert(Daughters_pdg.end(), temp_Daughters_pdg.begin(), temp_Daughters_pdg.end());
            Daughters_eta.insert(Daughters_eta.end(), temp_Daughters_eta.begin(), temp_Daughters_eta.end());
            Daughters_phi.insert(Daughters_phi.end(), temp_Daughters_phi.begin(), temp_Daughters_phi.end());
            Daughters_charge.insert(Daughters_charge.end(), temp_Daughters_charge.begin(), temp_Daughters_charge.end());
            Daughters_flag.insert(Daughters_flag.end(), temp_Daughters_flag.begin(), temp_Daughters_flag.end());
			Daughters_flav.insert(Daughters_flav.end(), temp_Daughters_flav.begin(), temp_Daughters_flav.end());
			nd = nPack;
			if(hadPDG==1) nd_b = nd;
			if(hadPDG==2) nd_d = nd;

			nDaughters.push_back(nd);
   			nDaughters_B.push_back(nd_b);
   			nDaughters_D.push_back(nd_d);

   		}
	} //if pdg

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
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_flag,
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
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_flag,
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
	tree->Branch("Daughters_flag", &Daughters_flag);
    tree->Branch("Daughters_flav", &Daughters_flav);        // flavor of the mother of the vertex
	tree->Branch("Daughters_pt", &Daughters_pt);        // from 0.8 on
    //tree->Branch("Daughters_pdg", &Daughters_pdg);
	tree->Branch("Daughters_eta", &Daughters_eta);
	tree->Branch("Daughters_phi", &Daughters_phi);      // ok
	tree->Branch("Daughters_charge", &Daughters_charge); // ok +-1
	
	tree->Branch("nTrks", &ntrks);                      // all the tracks in the event
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
    tree->Branch("trk_ipz", &trk_ipz);
    tree->Branch("trk_ipzsig", &trk_ipzsig);
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
        
    tree->Branch("preds", &preds);
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

//    std::ifstream file(genmatch_csv_);
//	if (!file.is_open()) {
//              std::cerr << "Failed to open file: " << genmatch_csv_ << std::endl;
//              return;
//         }
//
//	std::string line;
//	std::getline(file, line);  // Skip header
//	
//	int line_no = 1;
//	while (std::getline(file, line)) {
//	    ++line_no;
//	    if (line.empty()) continue;
//	
//	    std::stringstream ss(line);
//	    std::string run_str, lumi_str, evt_str, sig_str;
//	
//	    // Parse CSV fields (note: will fail if sig_str contains a comma inside quotes)
//	    if (!std::getline(ss, run_str, ',')) continue;
//	    if (!std::getline(ss, lumi_str, ',')) continue;
//	    if (!std::getline(ss, evt_str, ',')) continue;
//	    if (!std::getline(ss, sig_str)) continue;
//	
//	    // Strip potential quotes
//	    sig_str.erase(std::remove(sig_str.begin(), sig_str.end(), '"'), sig_str.end());
//	
//	    // Strip brackets
//	    sig_str.erase(std::remove(sig_str.begin(), sig_str.end(), '['), sig_str.end());
//	    sig_str.erase(std::remove(sig_str.begin(), sig_str.end(), ']'), sig_str.end());
//	
//	    try {
//	        unsigned int run = std::stoul(run_str);
//	        unsigned int lumi = std::stoul(lumi_str);
//	        unsigned int evt = std::stoul(evt_str);
//	
//	        std::vector<int> indices;
//	        std::stringstream sig_ss(sig_str);
//	        std::string val;
//	        while (std::getline(sig_ss, val, ',')) {
//	            if (!val.empty())
//	                indices.push_back(std::stoi(val));
//	        }
//	
//	        sigMatchMap_[{run, lumi, evt}] = indices;
//	
//	    } catch (const std::exception& e) {
//	        std::cerr << "Failed to parse line " << line_no << ": " << e.what() << "\nLine: " << line << std::endl;
//	    }
//	}
   	
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

