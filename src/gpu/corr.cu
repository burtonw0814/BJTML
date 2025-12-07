/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"

/*Grayscale Colors*/
#include "gpu/pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "gpu/cuda_launch_parameters.h"

/*Kernels*/
__global__ void Corr__ResetCorrScoresKernel(float* dev_corr_score, float* dev_corr_norm) {
	//dev_intersection_score[0] = 0;
	//dev_union_score[0] = 0;
	dev_corr_score[0] = 0.0;
	dev_corr_norm[0]  = 0.0;
}

__global__ void CorrKernel(unsigned char* dev_A, 
                           unsigned char* dev_B, 
                           unsigned char* dev_C, 
                           float* dev_corr_score, 
                           float* dev_corr_norm, 
                           int width, 
                           int height) {
                           
   // Rendered, original intensity, seg map
   
    // int* dev_intersection_score, int* dev_union_score,
    // int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width
    
	/*Global Thread*/
	//int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int pix_x_idx=blockIdx.x * blockDim.x + threadIdx.x; // pixel x dimension relative to (reduced resolution) cropped bounds
    int pix_y_idx=blockIdx.y * blockDim.y + threadIdx.y; // pixel y dimension relative to (reduced resolution) cropped bounds
    
    if (pix_x_idx<0)        {return;}
    if (pix_x_idx>=width)   {return;}
    if (pix_y_idx<0)        {return;}
    if (pix_y_idx>=height)  {return;}
    
    //printf("--Passed boundary check %d %d %d--",pix_x_idx, pix_y_idx, width, height);
    
    float seg_query=((float)dev_C[pix_y_idx*width+pix_x_idx]); //[ray_pt_x_idx*h_proj_map+ray_pt_y_idx]);///255.0f;
    if (seg_query==0) {return;}
    
    // Get means
    float proj_mean=0.0;
    float seg_mean=0.0;
            
    // Get patch wise means
    int r=17;
    float ct=0.0;
    for (int ii=-1*r; ii<r+1; ii++)     // ht
    {
        for (int jj=-1*r; jj<r+1; jj++) // wd
        {   
            int query_y_idx=pix_y_idx+ii;
            int query_x_idx=pix_x_idx+jj;
            
            if (query_y_idx>=0 && query_y_idx<height && query_x_idx>=0 && query_x_idx<width)
            {
                ct=ct+1;
                
                float seg_query=-1.0*((float)dev_B[query_y_idx*width+query_x_idx]);
                float proj_query=(float)dev_A[query_y_idx*width+query_x_idx];

                seg_mean=seg_mean+seg_query;
                proj_mean=proj_mean+proj_query;
            }
        }
    }
    
    // Get numerator and denominator values
    seg_mean=seg_mean/ct;
    proj_mean=proj_mean/ct;

    // Get numerator
    float num=0.0;
    float den0=0.0;
    float den1=0.0;
    float proj_var=0.0;
    float seg_var=0.0;   
    float ctt=0.0; 
    
    for (int ii=-1*r; ii<r+1; ii++)     // ht
    {
        for (int jj=-1*r; jj<r+1; jj++) // wd
        {
            int query_y_idx=pix_y_idx+ii;
            int query_x_idx=pix_x_idx+jj;

            if (query_y_idx>=0 && query_y_idx<height && query_x_idx>=0 && query_x_idx<width)
            {
                
                float seg_query=-1.0*((float)dev_B[query_y_idx*width+query_x_idx]); ///255.0f
                float proj_query=(float)dev_A[query_y_idx*width+query_x_idx];

                proj_var=proj_var+(proj_query-proj_mean)*(proj_query-proj_mean);
                seg_var=seg_var+(seg_query-seg_mean)*(seg_query-seg_mean);
                ctt++;
                
                num=num+(seg_query-seg_mean)*(proj_query-proj_mean);
            
                den0=den0+(seg_query-seg_mean)*(seg_query-seg_mean);
                den1=den1+(proj_query-proj_mean)*(proj_query-proj_mean);
            }
        }
    }

    float metric=num/(sqrt(den0)*sqrt(den1)+0.1);
    //metric_map_device[my_split*n_proj_map+ray_pt_x_idx*h_proj_map+ray_pt_y_idx]=wt*metric; // pix_idx  (proj_var/ctt)*
    atomicAdd(&dev_corr_score[0], metric);
    atomicAdd(&dev_corr_norm[0],  1);
    
	// Compute patch mean
	
	// Compute patch stdv
	
	// Convert to Subsize
	//i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
	//	diff_kernel_left_x;
		
	// dev_A and dev_B are row major, NOT column major!
	
	// Unpack x, y coords, perform local correlation for each patch, check bounds
    
    /*
	// Convert to Subsize
	i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
		diff_kernel_left_x;
    
	// If Correct Width and Height
	if (i < width * height) {
		int A_element = dev_A[i];
		int B_element = dev_B[i];

		if (A_element > 0 || B_element > 0) {
			atomicAdd(&dev_union_score[0], 1);
			if (A_element > 0 && B_element > 0) {
				atomicAdd(&dev_intersection_score[0], 1);
			}
		}
	}
	*/
    
}

namespace gpu_cost_function {
	double GPUMetrics::Corr(GPUImage* image_A, GPUImage* image_B, GPUImage* image_C) {
	    

	    // Rendered, original intensity, seg map
        
		// Extract Bounding Boxes
		//int* bounding_box_A = image_A->GetBoundingBox();
		//int* bounding_box_B = image_B->GetBoundingBox();
        
		// ASSUMES IMAGES ARE SAME DIMENSION!!!!!!
		int height = image_A->GetFrameHeight();
		int width =  image_A->GetFrameWidth();
		
		// Reset the IOU Score
		//IOU__ResetIOUScoresKernel<< <1, 1 >> >(dev_intersection_score_, dev_union_score_);
		Corr__ResetCorrScoresKernel<< <1, 1 >> >(dev_corr_score_, dev_corr_norm_);
        
		// Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges./
		//int diff_kernel_left_x = max(min(bounding_box_A[0], bounding_box_B[0]), 0);
		//int diff_kernel_bottom_y = max(min(bounding_box_A[1], bounding_box_B[1]), 0);
		//int diff_kernel_right_x = min(max(bounding_box_A[2], bounding_box_B[2]), width - 1);
		//int diff_kernel_top_y = min(max(bounding_box_A[3], bounding_box_B[3]), height - 1);
		//int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		//int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;
		
		int diff_kernel_left_x =          0;
		int diff_kernel_bottom_y =        0;
		int diff_kernel_right_x =         width-1;
		int diff_kernel_top_y =           height-1;
		int diff_kernel_cropped_width =   width;
		int diff_kernel_cropped_height =  height;
        
		//dim_grid_image_processing_ = dim3(
		//	ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
		//	ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));
		
		// Split over cam 0 ray pts
        int block_size0=32; // Max here is 32
        dim3 dim_block0(block_size0, block_size0, 1);
        dim3 dim_grid0((diff_kernel_cropped_width  + dim_block0.x - 1) / dim_block0.x, 
                       (diff_kernel_cropped_height + dim_block0.y - 1) / dim_block0.y);
        
		// IOU Kernel
		//IOUKernel << <dim_grid_image_processing_, threads_per_block >> >(
        //  image_A->GetDeviceImagePointer(), image_B->GetDeviceImagePointer(), dev_intersection_score_,
		//	dev_union_score_,
		//	width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
		
		CorrKernel<<<dim_grid0, dim_block0>>>(
                                                    image_A->GetDeviceImagePointer(), 
                                                    image_B->GetDeviceImagePointer(), 
                                                    image_C->GetDeviceImagePointer(), 
                                                    dev_corr_score_,
                                                    dev_corr_norm_,
                                                    width, height); // dim_grid_image_processing_,threads_per_block
        
		// Return IOU Score
		//cudaMemcpy(intersection_score_, dev_intersection_score_, sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(union_score_, dev_union_score_, sizeof(int), cudaMemcpyDeviceToHost);
		//return static_cast<double>(intersection_score_[0]) / static_cast<double>(union_score_[0]);
		
		cudaMemcpy(corr_score_, dev_corr_score_, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(corr_norm_,  dev_corr_norm_,  sizeof(float), cudaMemcpyDeviceToHost);
		
		//printf("-- %f %f--",corr_score_[0], corr_norm_[0]);
		
		return static_cast<double>(corr_score_[0]) / static_cast<double>(corr_norm_[0]);
		
		//return 0.0;
	}
}
