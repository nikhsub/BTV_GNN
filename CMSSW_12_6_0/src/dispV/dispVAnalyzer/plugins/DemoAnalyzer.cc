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

DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig):

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
	PupInfoT_ (consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("addPileupInfo")))
{
	edm::Service<TFileService> fs;	
	//usesResource("TFileService");
   	tree = fs->make<TTree>("tree", "tree");
}

DemoAnalyzer::~DemoAnalyzer() {
}

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

  njets.clear();
  jet_pt.clear();
  jet_eta.clear();
  jet_phi.clear();

  nSVs.clear();
  SV_x.clear();
  SV_y.clear();
  SV_z.clear();
  SV_pt.clear();
  SV_mass.clear();
  SV_ntrks.clear();


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
   for (const auto& track : alltracks) {
	reco::TransientTrack t_trk = (*theB).build(track);
	if (!(t_trk.isValid())) continue;
	Measurement1D ip2d = IPTools::signedTransverseImpactParameter(t_trk, direction, pv).second;
	Measurement1D ip3d = IPTools::signedImpactParameter3D(t_trk, direction, pv).second;	

   	trk_ip2d.push_back(ip2d.value());
	trk_ip3d.push_back(ip3d.value());
	trk_ip2dsig.push_back(ip2d.significance());
	trk_ip3dsig.push_back(ip3d.significance());
	trk_p.push_back(track.p());
	trk_pt.push_back(track.pt());
	trk_eta.push_back(track.eta());
	trk_phi.push_back(track.phi());
	trk_charge.push_back(track.charge());
	trk_nValid.push_back(track.numberOfValidHits());
	trk_nValidPixel.push_back(track.hitPattern().numberOfValidPixelHits());
	trk_nValidStrip.push_back(track.hitPattern().numberOfValidStripHits());
	ntrk++;
   }
   ntrks.push_back(ntrk);

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
   }
   nSVs.push_back(nsvs);


   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginJob() {
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

	tree->Branch("nJets", &njets);
	tree->Branch("jet_pt", &jet_pt);
        tree->Branch("jet_eta", &jet_eta);
        tree->Branch("jet_phi", &jet_phi);

	tree->Branch("nSVs", &nSVs);
	tree->Branch("SV_x", &SV_x);
	tree->Branch("SV_y", &SV_y);
	tree->Branch("SV_z", &SV_z);
	tree->Branch("SV_pt", &SV_pt);
	tree->Branch("SV_mass", &SV_mass);
	tree->Branch("SV_ntrks", &SV_ntrks);

}


// ------------ method called once each job just after ending the event loop  ------------
void DemoAnalyzer::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DemoAnalyzer);
