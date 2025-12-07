cmake_minimum_required(VERSION 3.19)
project(Biplanar-Joint-Track-Machine-Learning
        LANGUAGES CXX CUDA)


# 1. Settings
set(CMAKE_CXX_STANDARD 17)
set(USE_CUDNN 1)
set(QT_VERSION 5)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
   
          
# 2. Set dirs so cmake can find external packages  
set(Qt5_DIR "/usr/local/Qt-5.15.2/lib/cmake/Qt5")#"/usr/local/Qt-5.15.2")
set(VTK_DIR "/home/will/vtk/build/")
set(Torch_DIR "/home/will/libtorch/")
#set(Torch_DIR "/home/will/libtorch/share/cmake/Torch/")
set(OpenCV_DIR "/home/will/opencv/build")


# 3. Find Required Packages
find_package(VTK 9 REQUIRED COMPONENTS
            CommonCore
            RenderingCore
            RenderingOpenGL2
            IOCore
            IOImage
            IOGeometry
            GUISupportQt
            ChartsCore
            ViewsContext2D
            RenderingAnnotation
            InteractionStyle
            vtksys
)
#find_package(VTK 9 REQUIRED)
set(REQUIRED_LIBS Core Gui Widgets Sql Quick)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Sql Qt5::Quick)
find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)   


# 4. Help cmake find headers for external deps, if needed
include_directories($(VTK_INCLUDE_DIRS))
include_directories("./src/")     # Helps to find files
include_directories("./include/") # Helps to find files
include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#set(res ${PROJECT_SOURCE_DIR}/res)


# 5. Define dirs and headers for each module in the code base

# Header dir and include files for gui module
set(GUI_HEADER_DIR ${PROJECT_SOURCE_DIR}//include/gui)
file(GLOB GUI_HEADER_FILES ${GUI_HEADER_DIR}/*.ui ${GUI_HEADER_DIR}/*.h)

# Header dir and include files for gpu module
set(GPU_HEADER_DIR ${PROJECT_SOURCE_DIR}//include/gpu)
file(GLOB HEADER_FILES ${GPU_HEADER_DIR}/*.cuh ${GPU_HEADER_DIR}/*.h)

# Header dir and include files for core module
set(CORE_HEADER_DIR ${PROJECT_SOURCE_DIR}//include/core)
file(GLOB CORE_HEADER_FILES ${CORE_HEADER_DIR}/*.h)

# Header dir and include files for cost_functions
set(COST_HEADER_DIR ${PROJECT_SOURCE_DIR}//include/cost_functions)
file(GLOB HEADER_FILES ${COST_HEADER_DIR}/*.h)













 
# 6. Choose all relevant files to build, including source code (src) and headers (include)
#main.cpp # Add main.cpp in root dir for test case
#     main.cpp
#     my_cuda.cuh
#     my_cuda.cu
add_executable(${PROJECT_NAME} 
    
    ./src/gpu/dilate_edge_detected_image.cu
    ./src/gpu/edge_detect_rendered_implant_model.cu
    ./src/gpu/gpu_dilated_frame.cu
    ./src/gpu/gpu_edge_frame.cu
    ./src/gpu/gpu_frame.cu
    ./src/gpu/gpu_image.cu
    ./src/gpu/gpu_image_functions.cu
    ./src/gpu/gpu_intensity_frame.cu
    ./src/gpu/gpu_metrics.cu
    ./src/gpu/gpu_model.cu
    ./src/gpu/implant_mahfouz_metric.cu
    ./src/gpu/iou.cu
    ./src/gpu/corr.cu
    ./src/gpu/l_1_1_matrix_diff_norm.cu
    ./src/gpu/metric_toolbox.cu
    ./src/gpu/pose_matrix.cpp
    ./src/gpu/registration_metric.cu
    ./src/gpu/render_drr_engine.cu
    ./src/gpu/render_engine.cu
    ./src/gpu/distance_map_metric.cu
    ./src/gpu/fast_implant_dilation_metric.cu
    ./src/gpu/gpu_heatmaps.cu
    ./src/gpu/curvature_hausdorf_metric.cu
    ${GPU_HEADER_FILES}
    
    ./src/cost_functions/CostFunction.cpp
	./src/cost_functions/CostFunctionManager.cpp
	./src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp
	./src/cost_functions/DIRECT_DILATION.cpp
	./src/cost_functions/DIRECT_DILATION_POLE_CONSTRAINT.cpp
	./src/cost_functions/DIRECT_DILATION_SAME_Z.cpp
	./src/cost_functions/DIRECT_DILATION_T1.cpp
	./src/cost_functions/DIRECT_MAHFOUZ.cpp
	./src/cost_functions/sym_trap_function.cpp
	./src/cost_functions/IoU.cpp
	./src/cost_functions/Corr.cpp
	${COST_HEADER_FILES}
	
	./src/core/data_structures_6D.cpp
	./src/core/direct_data_storage.cpp
	./src/core/frame.cu
	./src/core/location_storage.cpp
	./src/core/model.cpp
	./src/core/optimizer_manager.cpp
	./src/core/optimizer_settings.cpp
	./src/core/stl_reader.cpp
	./src/core/STLReader.cpp
	./src/core/machine_learning_tools.cpp
	./src/core/sym_trap_functions.cpp
	./src/core/ambiguous_pose_processing.cpp
	./src/core/curvature_utilities.cpp
	${CORE_HEADER_FILES}
	
	./src/gui/about.cpp
	./src/gui/controls.cpp
	./src/gui/drr_tool.cpp
	./src/gui/main.cpp
	./src/gui/mainscreen.cpp
	./src/gui/settings_control.cpp
	./src/gui/viewer.cpp
	${GUI_HEADER_FILES}
	${GUI_HEADER_DIR}/mainscreen.qrc
    )
        
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# 7. Enables CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 8. Link external deps
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} /usr/local/cuda/lib64/libcurand.so)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES} VTK::CommonCore
                                                       VTK::RenderingCore
                                                       VTK::RenderingOpenGL2
                                                       VTK::IOCore
                                                       VTK::IOImage
                                                       VTK::IOGeometry
                                                       VTK::GUISupportQt
                                                       VTK::ChartsCore
                                                       VTK::ViewsContext2D
                                                       VTK::RenderingAnnotation
                                                       VTK::InteractionStyle
                                                       VTK::vtksys)

# 9. VTK boilerplate code
vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
)







