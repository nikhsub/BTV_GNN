#pragma once
#include <vector>
#include <utility>

std::pair<std::vector<int>, std::vector<float>> matchHadronsToSV(
    std::vector<std::vector<float>> distances,
    const std::vector<float>& SVtrk_pt,
    const std::vector<float>& SVtrk_eta,
    const std::vector<float>& SVtrk_phi,
    const std::vector<int>& SVtrk_SVIdx,
    const std::vector<float>& Daughters_pt,
    const std::vector<float>& Daughters_eta,
    const std::vector<float>& Daughters_phi,
    const std::vector<int>& Daughters_flag,
    int n_Hadrons
);

void printDistanceMatrix(const std::vector<std::vector<float>>& distances);