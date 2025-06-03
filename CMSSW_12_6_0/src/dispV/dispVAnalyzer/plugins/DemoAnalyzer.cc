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
#include <iostream>
#include <omp.h>
#include <unordered_set>

DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig, const ONNXRuntime *cache):

	theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
	TrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
	PVCollT_ (consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("primaryVertices"))),
	SVCollT_ (consumes<edm::View<reco::VertexCompositePtrCandidate>>(iConfig.getUntrackedParameter<edm::InputTag>("secVertices"))),
  	LostTrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("losttracks"))),
	jet_collT_ (consumes<edm::View<reco::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
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
{
	edm::Service<TFileService> fs;	
	//usesResource("TFileService");
   	tree = fs->make<TTree>("tree", "tree");
}

DemoAnalyzer::~DemoAnalyzer() {
}

std::unique_ptr<ONNXRuntime> DemoAnalyzer::initializeGlobalCache(const edm::ParameterSet &iConfig) {
    return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DemoAnalyzer::globalEndJob(const ONNXRuntime *cache) {}

std::optional<std::tuple<float, float, float>> DemoAnalyzer::isAncestor(const reco::Candidate* ancestor, const reco::Candidate* particle)
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
}

int DemoAnalyzer::checkPDG(int abs_pdg)
{
	std::vector<int> pdgList_B = { 521, 511, 531, 541, //Bottom mesons
				       5122, 5112, 5212, 5222, 5132, 5232, 5142, 5332, 5142, 5242, 5342, 5512, 5532, 5542, 5554}; //Bottom Baryons

	std::vector<int> pdgList_D = {411, 421, 431,      // Charmed mesons
                                     4122, 4222, 4212, 4112, 4232, 4132, 4332, 4412, 4422, 4432, 4444}; //Charmed Baryons

	if(std::find(pdgList_B.begin(), pdgList_B.end(), abs_pdg) != pdgList_B.end()){
		return 1;
	}
	else if(std::find(pdgList_D.begin(), pdgList_D.end(), abs_pdg) != pdgList_D.end()){
	       	return 2;
	}
	else{
		return 0;
	}

}

bool DemoAnalyzer::hasDescendantWithId(const reco::Candidate* particle, const std::vector<int>& pdgIds)
{
    // Base case: If the particle is null, return false
    if (!particle) {
        return false;
    }

    // Loop over all daughters
    for (size_t i = 0; i < particle->numberOfDaughters(); i++) {
        const reco::Candidate* daughter = particle->daughter(i);
        
        // Check if the current daughter is in the D hadron list
        if (daughter && std::find(pdgIds.begin(), pdgIds.end(), daughter->pdgId()) != pdgIds.end()) {
            return true; // Found a D hadron anywhere in the decay chain
        }

        // Recursively check deeper in the decay chain
        if (hasDescendantWithId(daughter, pdgIds)) {
            return true;
        }
    }

    return false; // No D hadron found in the decay chain
}

bool DemoAnalyzer::isGoodVtx(TransientVertex& tVTX){

   reco::Vertex tmpvtx(tVTX);
   return (tVTX.isValid() &&
    !tmpvtx.isFake() &&
    (tmpvtx.nTracks(vtxweight_)>1) &&
    (tmpvtx.normalizedChi2()>0) &&
    (tmpvtx.normalizedChi2()<10));
}

std::vector<TransientVertex> DemoAnalyzer::TrackVertexRefit(std::vector<reco::TransientTrack> &Tracks, std::vector<TransientVertex> &VTXs){
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
          if(isGoodVtx(newsv)) newVTXs.push_back(newsv);
      }
  
  }
  return newVTXs;
}

void DemoAnalyzer::vertexMerge(std::vector<TransientVertex>& VTXs, double maxFraction, double minSignificance) {
    VertexDistance3D dist;
    std::vector<TransientVertex> cleaned;

    for (size_t i = 0; i < VTXs.size(); ++i) {
        if (!VTXs[i].isValid()) continue;

        bool shared = false;
        VertexState s1 = VTXs[i].vertexState();

        for (size_t j = 0; j < VTXs.size(); ++j) {
            if (i == j || !VTXs[j].isValid()) continue;

            VertexState s2 = VTXs[j].vertexState();

            // Custom shared fraction calculation (via momentum comparison)
            int sharedTracks = 0;
            const auto& tracks_i = VTXs[i].originalTracks();
            const auto& tracks_j = VTXs[j].originalTracks();

            for (const auto& ti : tracks_i) {
                for (const auto& tj : tracks_j) {
                    double dpt = std::abs(ti.track().pt() - tj.track().pt());
                    if (dpt < 1e-3) {
                        ++sharedTracks;
                        break;
                    }
                }
            }

            double sharedFrac = double(sharedTracks) / std::min(tracks_i.size(), tracks_j.size());

            double dist_sig = dist.distance(s1, s2).significance();


            if (sharedFrac > maxFraction && dist_sig < minSignificance) {
                shared = true;
                break;
            }
        }

        if (!shared) {
            cleaned.push_back(VTXs[i]);
        }
    }

    VTXs = cleaned;
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




//
// member functions
//

// ------------ method called for each event  ------------
void DemoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace pat;

  nPU = 0;

  Hadron_pt.clear();
  Hadron_eta.clear();
  Hadron_phi.clear();
  Hadron_GVx.clear();
  Hadron_GVy.clear();
  Hadron_GVz.clear(); 
  nHadrons.clear();
  nGV.clear();
  nGV_B.clear();
  nGV_D.clear();
  GV_flag.clear();
  nDaughters.clear();
  nDaughters_B.clear();
  nDaughters_D.clear();
  Daughters_flag.clear();
  Daughters_flav.clear();
  Daughters_pt.clear();
  Daughters_eta.clear();
  Daughters_phi.clear();
  Daughters_charge.clear();

  ntrks.clear();
  trk_ip2d.clear();
  trk_ip3d.clear();
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
  SVtrk_eta.clear();
  SVtrk_phi.clear();

  preds.clear();
  cut.clear();

  nSVs_reco.clear();
  SV_x_reco.clear();
  SV_y_reco.clear();
  SV_z_reco.clear();
  SV_chi2_reco.clear();


  Handle<PackedCandidateCollection> patcan;
  Handle<PackedCandidateCollection> losttracks;
  Handle<edm::View<reco::Jet> > jet_coll;
  Handle<edm::View<reco::GenParticle> > pruned;
  Handle<edm::View<pat::PackedGenParticle> > packed;
  Handle<edm::View<reco::GenParticle> > merged;
  Handle<reco::VertexCollection> pvHandle;
  Handle<edm::View<reco::VertexCompositePtrCandidate>> svHandle ;
  Handle<std::vector< PileupSummaryInfo > > PupInfo;

  std::vector<reco::Track> alltracks;

  iEvent.getByToken(TrackCollT_, patcan);
  iEvent.getByToken(LostTrackCollT_, losttracks);
  iEvent.getByToken(jet_collT_, jet_coll);
  iEvent.getByToken(prunedGenToken_,pruned);
  iEvent.getByToken(packedGenToken_,packed);
  iEvent.getByToken(mergedGenToken_, merged);
  iEvent.getByToken(PVCollT_, pvHandle);
  iEvent.getByToken(SVCollT_, svHandle);

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


  for (auto const& itrack : *patcan){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
	       alltracks.push_back(tmptrk);
           }
       }
   }

   for (auto const& itrack : *losttracks){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
               alltracks.push_back(itrack.pseudoTrack());
           }
       }
   }

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
       edge_index_flat_f.push_back(static_cast<float>(edge_j[k]));
   }

   std::vector<float> edge_attr_flat;
   edge_attr_flat.reserve(edge_features.size() * 10);
   for (const auto& feat : edge_features)
       edge_attr_flat.insert(edge_attr_flat.end(), feat.begin(), feat.end());

      
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

   for (size_t i = 0; i < logits_data.size(); ++i) {
       float logit = logits_data[i];
       if (std::abs(logit) > 40.0f) {
           std::cout << "=== Anomalous Logit ===" << std::endl;
           std::cout << "Index " << i << ", Logit: " << logit << std::endl;

           const auto& features = track_features[i];
           std::cout << "Track Features: [";
           for (size_t j = 0; j < features.size(); ++j) {
               std::cout << features[j];
               if (j != features.size() - 1) std::cout << ", ";
           }
           std::cout << "]\n";
       }
   }

   std::vector<reco::TransientTrack> t_trks_SV;

   std::vector<std::pair<float, size_t>> score_index_pairs;
   for (size_t i = 0; i < logits_data.size(); ++i) {
       if (std::isnan(logits_data[i])) {
           std::cerr << "Warning: NaN in logit at index " << i;
       }
       std::cout << "Logit" << i << logits_data[i] << std::endl;
       float raw_score = sigmoid(logits_data[i]);
       if (std::isnan(raw_score)) {
           std::cerr << "Warning: NaN in sigmoid at index " << i << ", setting to 0.0\n";
       }
       preds.push_back(raw_score);
       score_index_pairs.emplace_back(raw_score, i);  // (score, index)
   }
   
   // Step 2: Sort by descending score
   std::sort(score_index_pairs.begin(), score_index_pairs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
   
   size_t n_keep = std::max<size_t>(1, score_index_pairs.size() * TrackPredCut_);  // ensure at least one
   for (size_t k = 0; k < n_keep; ++k) {
       size_t idx = score_index_pairs[k].second;
       t_trks_SV.push_back(t_trks[idx]);
   }

   float event_cut_threshold = score_index_pairs[n_keep - 1].first;
   cut.push_back(event_cut_threshold);
   
   

   std::cout << "t_trks_SV size"<< t_trks_SV.size() << std::endl;

   std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters = clusterizer->clusters(pv, t_trks_SV);
   std::vector<TransientVertex> recoVertices;
   for (std::vector<TracksClusteringFromDisplacedSeed::Cluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); ++cluster) {
          if (cluster->tracks.size()<2) continue; 
          
          std::vector<TransientVertex> tmp_vertices = vtxmaker_.vertices(cluster->tracks);
          for (std::vector<TransientVertex>::iterator v = tmp_vertices.begin(); v != tmp_vertices.end(); ++v) {
           reco::Vertex tmpvtx(*v);
           if(v->isValid() &&
              !tmpvtx.isFake() &&
              (tmpvtx.nTracks(vtxweight_)>1) &&
              (tmpvtx.normalizedChi2()>0) &&
              (tmpvtx.normalizedChi2()<5)) recoVertices.push_back(*v); 

          }
    }

    //std::cout << "Recovertices size" << recoVertices.size() << std::endl;
   
   std::vector<TransientVertex> vertices = vtxmaker_.vertices(t_trks_SV);
   for(std::vector<TransientVertex>::iterator isv = vertices.begin(); isv!=vertices.end(); ++isv){
       if(!isGoodVtx(*isv)) isv = vertices.erase(isv)-1;

   }

   std::cout << "vertices size" << vertices.size() << std::endl;

   recoVertices.insert(recoVertices.end(), vertices.begin(), vertices.end());

   std::cout << "Recovertices size" << recoVertices.size() << std::endl;
	

   //std::vector<TransientVertex> newVTXs = recoVertices;
   std::vector<TransientVertex> newVTXs = TrackVertexRefit(t_trks_SV, recoVertices);
   std::cout << "newVTXs size" << newVTXs.size() << std::endl;
   vertexMerge(newVTXs, 0.3, 3.0);
   std::cout << "newVTXs size" << newVTXs.size() << std::endl;

   int nvtx=0;
   for(size_t ivtx=0; ivtx<newVTXs.size(); ivtx++){
   	reco::Vertex tmpvtx(newVTXs[ivtx]);
        nvtx++;
        SV_x_reco.push_back(tmpvtx.position().x());
        SV_y_reco.push_back(tmpvtx.position().y());
        SV_z_reco.push_back(tmpvtx.position().z());
        SV_chi2_reco.push_back(tmpvtx.normalizedChi2()); 
   }
   nSVs_reco.push_back(nvtx);

   int nhads = 0;
   int ngv = 0;
   int ngv_b = 0;
   int ngv_d = 0;
   int nd = 0;
   int nd_b = 0;
   int nd_d = 0;
   std::vector<float> temp_Daughters_pt;
   std::vector<float> temp_Daughters_eta;
   std::vector<float> temp_Daughters_phi;
   std::vector<int> temp_Daughters_charge;
   std::vector<int> temp_Daughters_flag;
   std::vector<int> temp_Daughters_flav;
   
   std::vector<int> pdgList_D = {411, 421, 431,      // Charmed mesons
                                 4122, 4222, 4212, 4112, 4232, 4132, 4332, 4412, 4422, 4432, 4444}; //Charmed Baryons   


   for(size_t i=0; i< merged->size();i++)
   { //prune loop
	temp_Daughters_pt.clear();
        temp_Daughters_eta.clear();
        temp_Daughters_phi.clear();
        temp_Daughters_charge.clear();
        temp_Daughters_flag.clear();
        temp_Daughters_flav.clear();
	const Candidate * prun_part = &(*merged)[i];
	if(!(prun_part->pt() > 10 && std::abs(prun_part->eta()) < 2.5)) continue;
	int hadPDG = checkPDG(std::abs(prun_part->pdgId()));
	int had_parPDG = checkPDG(std::abs(prun_part->mother(0)->pdgId()));
	if(hadPDG == 1 && hasDescendantWithId(prun_part, pdgList_D)) continue; //skipping B hadrons that decay to D hadrons
	if(hadPDG > 0 && !(hadPDG == had_parPDG))
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
			const Candidate * mother = pack;
                        if(mother != nullptr)
                        {
                                auto GV = isAncestor(prun_part, mother);
                                if(GV.has_value())
                                {
                                        std::tie(vx, vy, vz) = *GV;
					if (!std::isnan(vx) && !std::isnan(vy) && !std::isnan(vz)){
						nPack++;
						temp_Daughters_pt.push_back(pack->pt());
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
			        Hadron_GVx.push_back(vx);
                	        Hadron_GVy.push_back(vy); 
                	        Hadron_GVz.push_back(vz); 
                	        GV_flag.push_back(nhads-1); //Which hadron it belongs to
				addedGV = true;
			}
			Daughters_pt.insert(Daughters_pt.end(), temp_Daughters_pt.begin(), temp_Daughters_pt.end());
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
   nGV_D.push_back(ngv_d);


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
	}
   }
   nSVs.push_back(nsvs);


   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginStream(edm::StreamID) {
	tree->Branch("nPU", &nPU);
	tree->Branch("nHadrons", &nHadrons);
	tree->Branch("Hadron_pt", &Hadron_pt);
	tree->Branch("Hadron_eta", &Hadron_eta);
	tree->Branch("Hadron_phi", &Hadron_phi);
	tree->Branch("Hadron_GVx", &Hadron_GVx);
	tree->Branch("Hadron_GVy", &Hadron_GVy);
	tree->Branch("Hadron_GVz", &Hadron_GVz);
	tree->Branch("nGV", &nGV);
	tree->Branch("nGV_B", &nGV_B);
	tree->Branch("nGV_D", &nGV_D);
	tree->Branch("GV_flag", &GV_flag);
	tree->Branch("nDaughters", &nDaughters);
	tree->Branch("nDaughters_B", &nDaughters_B);
	tree->Branch("nDaughters_D", &nDaughters_D);
	tree->Branch("Daughters_flag", &Daughters_flag);
	tree->Branch("Daughters_flav", &Daughters_flav);
	tree->Branch("Daughters_pt", &Daughters_pt);
	tree->Branch("Daughters_eta", &Daughters_eta);
	tree->Branch("Daughters_phi", &Daughters_phi);
	tree->Branch("Daughters_charge", &Daughters_charge);
	
	tree->Branch("nTrks", &ntrks);
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
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
         
        tree->Branch("trk_i", &trk_i);
	tree->Branch("trk_j", &trk_j);
	tree->Branch("deltaR", &deltaR);
	tree->Branch("dca", &dca);
	tree->Branch("dca_sig", &dca_sig);
        tree->Branch("cptopv", &cptopv);
	tree->Branch("pvtoPCA_i", &pvtoPCA_i);
	tree->Branch("pvtoPCA_j", &pvtoPCA_j);
	tree->Branch("dotprod_i", &dotprod_i);
	tree->Branch("dotprod_j", &dotprod_j);
	tree->Branch("pair_mom", &pair_mom);
	tree->Branch("pair_invmass", &pair_invmass);
	

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
	tree->Branch("SVtrk_eta", &SVtrk_eta);
	tree->Branch("SVtrk_phi", &SVtrk_phi);
        
        tree->Branch("preds", &preds);
	tree->Branch("cut_val", &cut);

	tree->Branch("nSVs_reco", &nSVs_reco);
        tree->Branch("SV_x_reco", &SV_x_reco);
        tree->Branch("SV_y_reco", &SV_y_reco);
        tree->Branch("SV_z_reco", &SV_z_reco);
        tree->Branch("SV_chi2_reco", &SV_chi2_reco);

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
