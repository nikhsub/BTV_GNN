#include <vector>
#include <utility>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "DataFormats/Math/interface/deltaR.h"
#include "dispV/dispVAnalyzer/interface/matchedHadronsToSV.h"

// function for debugging
void printDistanceMatrix(const std::vector<std::vector<float>>& distances) {
    size_t nSV = distances.size();
    if (nSV == 0) return;

    size_t nGV = distances[0].size();

    //std::cout << "\n Distance Matrix (SV rows × GV cols):\n\n";

    // Print header row
    std::cout << std::setw(8) << "SV/GV";
    for (size_t j = 0; j < nGV; ++j) {
        std::cout << std::setw(10) << "GV[" + std::to_string(j) + "]";
    }
    std::cout << "\n";

    // Print matrix values
    for (size_t i = 0; i < nSV; ++i) {
        std::cout << std::setw(8) << "SV[" + std::to_string(i) + "]";
        for (size_t j = 0; j < nGV; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << distances[i][j];
        }
        std::cout << "\n";
    }
}


std::pair<std::vector<int>, std::vector<float>>  matchHadronsToSV(
    std::vector<std::vector<float>> distances,
    const std::vector<float>& SVtrk_pt,
    const std::vector<float>& SVtrk_eta,
    const std::vector<float>& SVtrk_phi,
    const std::vector<int>& SVtrk_SVIdx,
    const std::vector<float>& Daughters_pt,
    const std::vector<float>& Daughters_eta,
    const std::vector<float>& Daughters_phi,
    const std::vector<int>& Daughters_flag, // hadron index per daughter
    int n_Hadrons
) {
    //using namespace std;
    using namespace reco;
    // Returns
    // Hadron_SVIdx = Array of lenght = Hadron_pt.size() with index of the SV
    size_t nSV = distances.size(); //distances is nSV x nHadron matrix (the first dimension is nSV)
    std::vector<int> Hadron_SVIdx(n_Hadrons, -1); // Output
    std::vector<float> Hadron_SVDistance(n_Hadrons, -1); // Output
    //std::cout<<"Definitions here"<<"\n";

    while (true) {
        float minDist = 999.0;
        int bestSV = -1;
        int bestHad = -1;

        // Find minimum distance in current matrix
        // store in 
        // - minDist
        // - bestSV
        // - bestHad
        //std::cout<<nSV<<" "<<n_Hadrons<<"\n";
        for (size_t sv = 0; sv < nSV; ++sv) {
            for (int had = 0; had < n_Hadrons; ++had) {
                //std::cout<<"i, j"<<sv<<" "<<had<<"\n";
                if (distances[sv][had] < minDist) {
                    minDist = distances[sv][had];
                    bestSV = sv;
                    bestHad = had;
                }
            }
        }
        //std::cout<<"minDist "<<minDist<<"\n";
        //printDistanceMatrix(distances);
        if (minDist >= 997.0) break;  // done

        // Select tracks from SV
        // svTrackIdxs_fromBestSV is initialized every time
        // it stores the index of the tracks which originate from SV candidate in this loop
        std::vector<size_t> svTrackIdxs_fromBestSV;
        for (size_t i = 0; i < SVtrk_SVIdx.size(); ++i) {
            // among all tracks from all SV, select those from the candidate SV
            if (SVtrk_SVIdx[i] == bestSV && SVtrk_pt[i] > 0.8 && std::abs(SVtrk_eta[i]) < 2.5) {
                svTrackIdxs_fromBestSV.push_back(i);
            }
        }
        //std::cout<<"Here here"<<"\n";
        // Select daughters of Hadron
        std::vector<size_t> GenDaughtersIdxs_fromBestHad;
        for (size_t i = 0; i < Daughters_flag.size(); ++i) {
            if (Daughters_flag[i] == bestHad) {
                GenDaughtersIdxs_fromBestHad.push_back(i);
            }
        }
        //std::cout<<"Here here2"<<"\n";
        // Match logic: check for 1 (2) or more matched tracks by ΔR & dPt/pT
        int nRequiredCommonTracks = 2;
        int common = 0;
        for (size_t iSV : svTrackIdxs_fromBestSV) {
            //std::cout<<"here3 "<<iSV<<std::endl;
            for (size_t iHad : GenDaughtersIdxs_fromBestHad) {
                //std::cout<<"here4 "<<iHad<<std::endl;
                //std::cout<<SVtrk_eta.size() << " " << SVtrk_phi.size() <<std::endl;
                //std::cout<<SVtrk_eta[iSV] << "  " << SVtrk_phi[iSV]<<std::endl;
                //std::cout<<Daughters_eta[iHad] << "  " << Daughters_phi[iHad]<<std::endl;
                float dR = reco::deltaR(SVtrk_eta[iSV], SVtrk_phi[iSV], Daughters_eta[iHad], Daughters_phi[iHad]);
                //std::cout<<"dR "<<dR<<std::endl;
                float relPt = std::fabs(SVtrk_pt[iSV] - Daughters_pt[iHad]) / Daughters_pt[iHad];
                //std::cout<<"relPt "<<relPt<<std::endl;
                if (dR < 0.03 && relPt < 0.5) {
                    ++common;
                    if (common >= nRequiredCommonTracks) break; // break the iHad cycle
                }
            }
            if (common >= nRequiredCommonTracks) break; // break the iSV cycle
        }

        if (common >= nRequiredCommonTracks) {
            Hadron_SVIdx[bestHad] = bestSV;
            Hadron_SVDistance[bestHad]= minDist;
            for (int h = 0; h < n_Hadrons; ++h) distances[bestSV][h] = 999.0;  // remove SV row
            for (size_t s = 0; s < nSV; ++s) distances[s][bestHad] = 999.0;   // remove Hadron column
            //std::cout << " Matched Hadron[" << bestHad << "] to SV[" << bestSV << "] (distance = " << minDist << ", common tracks = " << common << ")\n";
        } else {
            distances[bestSV][bestHad] = 998.0;  // exclude this pair
        }
    }

    return std::make_pair(Hadron_SVIdx, Hadron_SVDistance);
}


