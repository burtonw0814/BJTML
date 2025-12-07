#include <iostream>
#include <string>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkAlgorithm.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

#include <opencv2/opencv.hpp>

#include "my_cuda.cuh"

int main(int argc, char* argv[]) {

    CUDA_Test* my_test= new CUDA_Test();
    my_test->test();

    /*
    std::cout << "Hello" << "\n";
    //vtkSmartPointer<vtkRenderer> renderer_=vtkSmartPointer<vtkRenderer>::New();
    
    std::string my_path="/home/will/Desktop/research/BJTML/BJTML/example_studies/Kneel_1/AT_K1_V1_0160.tif";
    cv::Mat my_im=cv::imread(my_path, cv::IMREAD_COLOR);
    cv::imshow("this->imd0", my_im);
    cv::waitKey(0);
    */
    
    
   
     
}


