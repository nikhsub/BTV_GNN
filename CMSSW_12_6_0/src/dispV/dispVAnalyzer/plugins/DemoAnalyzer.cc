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
  	LostTrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("losttracks"))),
	jet_collT_ (consumes<edm::View<reco::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
	prunedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("pruned"))),
  	packedGenToken_(consumes<edm::View<pat::PackedGenParticle> >(iConfig.getParameter<edm::InputTag>("packed"))),
	TrackPtCut_(iConfig.getUntrackedParameter<double>("TrackPtCut"))
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

bool DemoAnalyzer::checkPDG(int abs_pdg)
{
	std::vector<int> pdgList = { 521, 511, 531, 541, //Bottom mesons
			 	     411, 421, 431,      // Charmed mesons
			     	     4122, 4222, 4212, 4112, 4232, 4132, 4332, 4412, 4422, 4432, 4444,
				     5122, 5112, 5212, 5222, 5132, 5232, 5142, 5332, 5142, 5242, 5342, 5512, 5532, 5542, 5554};

	if(std::find(pdgList.begin(), pdgList.end(), abs_pdg) != pdgList.end()){
		return true;
	}
	else{
	       	return false;
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

  Hadron_pt.clear();
  Hadron_eta.clear();
  Hadron_phi.clear();
  Hadron_SVx.clear();
  Hadron_SVy.clear();
  Hadron_SVz.clear(); 
  nHadrons.clear();
  nSV.clear();
  Daughter1_pt.clear();
  Daughter1_eta.clear();
  Daughter1_phi.clear();
  Daughter1_charge.clear();
  Daughter2_pt.clear();
  Daughter2_eta.clear();
  Daughter2_phi.clear();
  Daughter2_charge.clear();

  ntrks.clear();
  trk_ip2d.clear();
  trk_ip3d.clear();
  trk_ip2dsig.clear();
  trk_ip3dsig.clear();
  trk_pt.clear();
  trk_eta.clear();
  trk_phi.clear();

  njets.clear();
  jet_pt.clear();
  jet_eta.clear();
  jet_phi.clear();


  Handle<PackedCandidateCollection> patcan;
  Handle<PackedCandidateCollection> losttracks;
  Handle<edm::View<reco::Jet> > jet_coll;
  Handle<edm::View<reco::GenParticle> > pruned;
  Handle<edm::View<pat::PackedGenParticle> > packed;
  Handle<reco::VertexCollection> pvHandle;

  std::vector<reco::Track> alltracks;

  iEvent.getByToken(TrackCollT_, patcan);
  iEvent.getByToken(LostTrackCollT_, losttracks);
  iEvent.getByToken(jet_collT_, jet_coll);
  iEvent.getByToken(prunedGenToken_,pruned);
  iEvent.getByToken(packedGenToken_,packed);
  iEvent.getByToken(PVCollT_, pvHandle);

  const auto& theB = &iSetup.getData(theTTBToken);
  reco::Vertex pv = (*pvHandle)[0];

  GlobalVector direction(1,0,0);
  direction = direction.unit();

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

   int njet = 0;
   for (auto const& ijet: *jet_coll){
   	jet_pt.push_back(ijet.pt());
	jet_eta.push_back(ijet.eta());
	jet_phi.push_back(ijet.phi());
	njet++;
   }
   njets.push_back(njet);
   
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
	trk_pt.push_back(track.pt());
	trk_eta.push_back(track.eta());
	trk_phi.push_back(track.phi());
	ntrk++;
   }
   ntrks.push_back(ntrk);

   int nhads = 0;
   int nsv = 0;
   for(size_t i=0; i< pruned->size();i++)
   { //prune loop
	const Candidate * prun_part = &(*pruned)[i];
	if(!(prun_part->pt() > 10 && std::abs(prun_part->eta()) < 2.5)) continue;
	if(checkPDG(std::abs(prun_part->pdgId())))
	{ //if pdg
		nhads++;
		Hadron_pt.push_back(prun_part->pt());
		Hadron_eta.push_back(prun_part->eta());
		Hadron_phi.push_back(prun_part->phi());
                for(size_t j=0; j< packed->size();j++)
		{ //packed outer
			bool tobreak = false;
			const Candidate *pack1 =  &(*packed)[j];
			if(!(pack1->pt() > 1 && std::abs(pack1->eta()) < 2.5 && std::abs(pack1->charge()) > 0)) continue;
			const Candidate * mother1 = pack1->mother(0);
			float vx1 = std::numeric_limits<float>::quiet_NaN();
    			float vy1 = std::numeric_limits<float>::quiet_NaN();
    			float vz1 = std::numeric_limits<float>::quiet_NaN();
			if(mother1 != nullptr)
			{
				auto SV1 = isAncestor(prun_part, mother1);
				if(SV1.has_value())
				{
					std::tie(vx1, vy1, vz1) = *SV1;
				}
			}
			for(size_t k=0; k<packed->size(); k++)
			{ //Packed inner
				if(j==k) continue;
				const Candidate *pack2 =  &(*packed)[k];
				if(!(pack2->pt() > 1 && std::abs(pack2->eta()) < 2.5 && std::abs(pack2->charge()) > 0)) continue;
				const Candidate * mother2 = pack2->mother(0);
                                if(mother2 !=nullptr){
					auto SV2 = isAncestor(prun_part, mother2);
					if(SV2.has_value()){
						auto [vx2, vy2, vz2] = *SV2;
						if (!std::isnan(vx1) && !std::isnan(vy1) && !std::isnan(vz1) && !std::isnan(vx2) && !std::isnan(vy2) && !std::isnan(vz2)) 
						{
							nsv++;
							Daughter1_pt.push_back(pack1->pt());
							Daughter2_pt.push_back(pack2->pt());
                                                        Daughter1_eta.push_back(pack1->eta());
                                                        Daughter2_eta.push_back(pack2->eta());
                                                        Daughter1_phi.push_back(pack1->phi());
                                                        Daughter2_phi.push_back(pack2->phi());
                                                        Daughter1_charge.push_back(pack1->charge());
                                                        Daughter2_charge.push_back(pack2->charge());
							Hadron_SVx.push_back(vx1);
							Hadron_SVy.push_back(vy1); //Here each vertex coordinate will correspond to the vertex of the had
							Hadron_SVz.push_back(vz1); //desc it thinks is coming from this corresponding daugther in the loop
							//Hadron_SVx.push_back(vx2);
                                                        //Hadron_SVy.push_back(vy2); 
							//Hadron_SVz.push_back(vz2);
							tobreak = true;
							break;
						}
					}
				}
			} //Packed inner
			if(tobreak) break;
		} //packed outer
	} //if pdg

   } //prune loop
   nHadrons.push_back(nhads);
   nSV.push_back(nsv);

   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginJob() {
	tree->Branch("nHadrons", &nHadrons);
	tree->Branch("Hadron_pt", &Hadron_pt);
	tree->Branch("Hadron_eta", &Hadron_eta);
	tree->Branch("Hadron_phi", &Hadron_phi);
	tree->Branch("Hadron_SVx", &Hadron_SVx);
	tree->Branch("Hadron_SVy", &Hadron_SVy);
	tree->Branch("Hadron_SVz", &Hadron_SVz);
	tree->Branch("nSV", &nSV);
	tree->Branch("Daughter1_pt", &Daughter1_pt);
	tree->Branch("Daughter1_eta", &Daughter1_eta);
	tree->Branch("Daughter1_phi", &Daughter1_phi);
	tree->Branch("Daughter1_charge", &Daughter1_charge);
	tree->Branch("Daughter2_pt", &Daughter2_pt);
        tree->Branch("Daughter2_eta", &Daughter2_eta);
        tree->Branch("Daughter2_phi", &Daughter2_phi);
        tree->Branch("Daughter2_charge", &Daughter2_charge);

	tree->Branch("nTrks", &ntrks);
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
	tree->Branch("trk_ip2dsig", &trk_ip2dsig);
        tree->Branch("trk_ip3dsig", &trk_ip3dsig);
	tree->Branch("trk_pt", &trk_pt);
	tree->Branch("trk_eta", &trk_eta);
	tree->Branch("trk_phi", &trk_phi);

	tree->Branch("nJets", &njets);
	tree->Branch("jet_pt", &jet_pt);
        tree->Branch("jet_eta", &jet_eta);
        tree->Branch("jet_phi", &jet_phi);

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
