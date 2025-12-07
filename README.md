# BJTML

 <img width="1844" height="1164" alt="BJTML_PIC" src="https://github.com/user-attachments/assets/08c61a45-bb31-426f-93bf-d0235bbf5fa4" />

 


 
 Extension of JTML software to be compatible with DU's biplanar tracking workflow.
 
 Single plane version:
 https://github.com/ajensen1234/JTML/
 
 Special thanks to Andrew Jensen, Paris Flood, and Scott Banks at UF for their hard work getting the single plane version up and running.


 Instructions:
 
 1. Install dependencies:
    
    Follow instructions at the following link to install dependencies (if it makes you want to pull your hair out then you're on the right track):
    https://github.com/burtonw0814/BJTML_link/blob/main/LINUX_BUILD.md
    
 3. Build using CMake:
    
    cmake --build .
    cmake CMakeLists.txt
    
 5. Run:
    
    ./Biplanar-Joint-Track-Machine-Learning


 Other notes:
 1. This repo only supports biplanar tracking. For single-plane tracking, see the single-plane predecessor repo referenced above.
 2. Camera calibration (intrinsics + extrinsics) is based on calibration data exported from XMA Lab (MayaCam2 format).
    Convert XMA Lab-based camcal data to the correct BJTML format using code in the MATLAB directory.
 3. Prior to registration, go to Settings->Optimization Settings and select IoU, IoU, and MASKED_CORRELATION as the loss functions for the Trunk, Branch, and Leaf stages, respectively. The other loss functions may not work well for the biplanar setting.
 4. Reliance on CNN-based segmentation may be avoided by selecting only MASKED_CORRELATION as the loss function for all three stages. However, the performance might then be less robust or require better initialization.
 5. Preliminary validation was performed by comparing auto-tracked TKA poses from BJTML to manually tracked values from a different software (Autoscoper, DSX, etc.). Sub-mm and sub-degree mean differences were observed between auto and manually tracked poses.
 6. Extension to other joints (e.g., hip) is possible by developing segmentation models for these joints, or by avoiding segmentation a la (4.) above.

