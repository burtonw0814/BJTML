// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/machine_learning_tools.h"





void tensor_prep(torch::Tensor& my_tens, bool norm_flag) {

    //
    if (norm_flag==true) {
        my_tens=my_tens.div(255.0f);
    }

    my_tens=torch::unsqueeze(my_tens, 0); // Unsqueeze channel dimension
    my_tens=torch::unsqueeze(my_tens, 0); // Unsqueeze batch dimension

}

void tensor_post(torch::Tensor& my_tens) {

    my_tens=torch::squeeze(my_tens, 0); // Squeeze batch dimension
    my_tens=torch::squeeze(my_tens, 0); // Squeeze channel dimension (because it is always 1)

    // Round to get class prediction
    my_tens=torch::round(my_tens);

    my_tens=my_tens.mul(255.0f);

}

cv::Mat mat_resize(cv::Mat& my_mat, int wd, int ht) {

    cv::Mat new_mat;

    cv::Size size(wd, ht);//the dst image size,e.g.100x100

    // Resize
    cv::resize(my_mat, new_mat, size, 0, 0, cv::INTER_LINEAR);

    return new_mat;
};

torch::Tensor mat_to_tensor(cv::Mat& my_im) {

    // IM is currently in 8 bit ints and is grey scale
    cv::Mat img_float2;

    // Convert Mat to float
    my_im.convertTo(img_float2, CV_32F);

    // Create tensor
    torch::Tensor out = torch::zeros({my_im.rows, my_im.cols});
    memcpy(out.data_ptr(), img_float2.data, out.numel() * sizeof(float));
    //std::cout << "Tensor sizes " << out.sizes() << "\n";
    
    // Create GPU tensor
    torch::Tensor gpu_tensor = out.to(torch::kCUDA); // Moved to GPU
    
    return gpu_tensor;
};

cv::Mat tensor_to_mat(torch::Tensor& my_tens) {

    // Tens is current in floats but has already been rounded
    // Also Tens must have shape (h, w)
    // Convert to ints
    // Create matrix
    // Convert tensor to uint
    my_tens=my_tens.to(torch::kByte).cpu().contiguous();

    //
    cv::Mat mr;
    auto yo = my_tens;
    uchar* data = (uint8_t*)(yo.data_ptr());
    mr = cv::Mat(my_tens.size(0), my_tens.size(1), CV_8UC1, data); //return only first channel
    //std::cout << "Mat sizes " << mr.rows << ", " << mr.cols << "\n";

    return mr;
};








cv::Mat segment_image(const cv::Mat& orig_image, bool black_sil_used,
                      torch::jit::Module* model, unsigned int input_width,
                      unsigned int input_height) {
                      
                               
        // Get original size of image to match segmentation map
        int wd_o=orig_image.cols;
        int ht_o=orig_image.rows;
        
        cv::Mat mat_in=orig_image.clone();
        
        //cv::imshow("this->imd0", mat_in);
        //cv::waitKey(0);

        // Resize
        std::vector<cv::Mat> mats_re;
        mats_re.push_back(mat_resize(mat_in, 1024, 1024));

        // Tensor creation
        std::vector<torch::Tensor> tens_vec;
        for (int i=0; i<mats_re.size(); i++) {
            tens_vec.push_back(mat_to_tensor(mats_re[i]));
        }

        // Additiional tensor prep
        for (int i=0; i<mats_re.size(); i++) {
            tensor_prep(tens_vec[i], true);
        }

        // Create jit format vector
        std::vector<torch::jit::IValue> inputs_vec;
        for (int i=0; i<tens_vec.size(); i++) {
            inputs_vec.push_back(torch::jit::IValue(tens_vec[i]));
        }

        // Execute the model and turn its output into a tensor.
        at::Tensor out_tens = model->forward(inputs_vec).toTensor();
        torch::Tensor out_tens_torch=torch::Tensor(out_tens);

        // Post-process tensor
        tensor_post(out_tens_torch);

        // Convert back to CV mat
        cv::Mat mat_out=tensor_to_mat(out_tens_torch);

        // Resize again
        mat_out=mat_resize(mat_out, wd_o, ht_o);
        
        //cv::imshow("this->imd0", mat_out);
        //cv::waitKey(0);
        
        return mat_out;
                      
                      
                      
          
          
    /*            
    std::cout << "Segmenting" << "\n";
                      
    //Create a GPU byte placeholder for memory purposes
    torch::Tensor gpu_byte_placeholder(
        torch::zeros({1, 1, input_height, input_width},
                     torch::device(torch::kCUDA).dtype(torch::kByte)));
    //Get the correct inversion for the image
    cv::Mat correct_inversion =
        (255 * black_sil_used) + ((1 - 2 * black_sil_used) * orig_image);
    cv::Mat padded;

    //Pad the image to a square based on the larger dimension
    if (correct_inversion.cols > correct_inversion.rows) {
        padded.create(correct_inversion.cols, correct_inversion.cols,
                      correct_inversion.type());
    } else {
        padded.create(correct_inversion.rows, correct_inversion.rows,
                      correct_inversion.type());
    }

    const unsigned int padded_width = padded.cols;
    const unsigned int padded_height = padded.rows;

    padded.setTo(cv::Scalar::all(0));

    //Copy things over to the GPU for the forward pass
    correct_inversion.copyTo(
        padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows)));
    cv::resize(padded, padded, cv::Size(input_width, input_height));
    cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
               input_height * input_width * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
               
    gpu_byte_placeholder=gpu_byte_placeholder.div(255.0f);

    //Define the machine learning inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(
        gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({2}));

    // Forward Pass and bring it back to host
    cudaMemcpy(padded.data,
               (255 * (model->forward(inputs).toTensor() > 0))
                   .to(torch::dtype(torch::kByte))
                   .flip({2})
                   .data_ptr(),
               input_height * input_width * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    cv::resize(padded, padded, cv::Size(padded_width, padded_height));
    cv::Mat unpadded =
        padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows));
        
    std::cout << "Done" << "\n";
    
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    minMaxLoc( unpadded, &minVal, &maxVal, &minLoc, &maxLoc );

    cout << "min val: " << minVal << endl;
    cout << "max val: " << maxVal << endl;

    return unpadded;
    */
    
}
