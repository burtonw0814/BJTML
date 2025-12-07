// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*DIRECT_DILATION Source*/
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "cost_functions/CostFunctionManager.h"

namespace jta_cost_function {
bool CostFunctionManager::initializeIoU(
    std::string &error_message) {
    /*Any cost function stage initialization proceedings go here.
    This is called when the optimizer begins a new stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    /*CUDA Error status container*/
    cudaError cudaStatus;
    
    error_message = cudaGetErrorString(cudaStatus);
    
    /*Return if success or not*/
    return true;
}
bool CostFunctionManager::destructIoU(std::string &error_message) {
    /*Any cost function stage initialization proceedings that involve
    creating new variables should be destructed here.
    This is called when the optimizer ends a stage.
    Must return whether or not the initialization was successful.
    To display an error message, simply store the message in
    the "error_message" variable and return false.*/

    return true;
}
double CostFunctionManager::costFunctionIoU() {
    /*Cost function implementation goes here.
    This procedure is called every time the optimizer wants to
    query the cost function at a given point.
    One must return this value as a double.*/
    
    gpu_principal_model_->RenderPrimaryCamera(
        gpu_principal_model_->GetCurrentPrimaryCameraPose());
        
    double metric_score;
    
    metric_score =
        gpu_metrics_->IOU(
             gpu_principal_model_->GetPrimaryCameraRenderedImage(),
             gpu_intensity_frames_A_->at(current_frame_index_)->GetInvertedGPUImage());
    
    double metric_score2;
    if (biplane_mode_) {
        //Render
        gpu_principal_model_->RenderSecondaryCamera(
            gpu_principal_model_->GetCurrentSecondaryCameraPose());

        metric_score2 =
        gpu_metrics_->IOU(
             gpu_principal_model_->GetSecondaryCameraRenderedImage(),
             gpu_intensity_frames_B_->at(current_frame_index_)->GetInvertedGPUImage());
                
    }
    
    return -1*(metric_score+metric_score2);//+metric_score2;
}
}  // namespace jta_cost_function
