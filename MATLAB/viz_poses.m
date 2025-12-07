clear all
close all
clc

% Import implant STLs and tracked poses from BJTML, visualize
% Will Burton

% Set up paths
data_root="/home/will/Desktop/research/BJTML/BJTML/example_studies/DU_VAL1/";
camcal_path1=data_root+"/DATA/Subject170/Camcal/Camcal0.txt";
camcal_path2=data_root+"/DATA/Subject170/Camcal/Camcal1.txt";
stl_path_fem=data_root+"/DATA/Subject170/STLs/femur.stl";
stl_path_tib=data_root+"/DATA/Subject170/STLs/tibia.stl";
poses_path_fem=data_root+"/DATA/Subject170/fem.txt";
poses_path_tib=data_root+"/DATA/Subject170/tib.txt";
poses_path_fem_out=data_root+"/DATA/Subject170/fem_out.txt";
poses_path_tib_out=data_root+"/DATA/Subject170/tib_out.txt";

% Import camcal in case we want to express in global cosys (BJTML poses are in cam0 cosys)
camcal1=get_camcal(camcal_path1);
camcal2=get_camcal(camcal_path2);

% Import STLs
stl_fem=stlread(stl_path_fem);
stl_tib=stlread(stl_path_tib);

% Import poses
xf=readmatrix(poses_path_fem); % Header lines are automatically skipped
xt=readmatrix(poses_path_tib);

% Export poses again for Python
writematrix(xf,poses_path_fem_out);
writematrix(xt,poses_path_tib_out);

% Visualize tracked pose

% Choose frame
f_idx=1; 

% Unpack femur pose
t_f=xf(f_idx,1:3)';
eul_f=xf(f_idx,4:6);
r_f=eul2rotm(eul_f*pi/180,"ZXY");

% Unpack tibia pose
t_t=xt(f_idx,1:3)';
eul_t=xt(f_idx,4:6);
r_t=eul2rotm(eul_t*pi/180,"ZXY");

% Apply poses to nodes
nf_tracked=(r_f*stl_fem.Points'+t_f)';
nt_tracked=(r_t*stl_tib.Points'+t_t)';

% Patch
figure;
p(1) = patch('Faces', stl_fem.ConnectivityList, 'Vertices', nf_tracked);
set(p(1),'FaceColor', 'r', 'EdgeColor','r','FaceAlpha',1,'EdgeAlpha',0,'SpecularStrength',.3);
hold on; axis equal;
p(1) = patch('Faces', stl_tib.ConnectivityList, 'Vertices', nt_tracked);
set(p(1),'FaceColor', 'c', 'EdgeColor','c','FaceAlpha',1,'EdgeAlpha',0,'SpecularStrength',.3);
light('Position',[1 1 1]*1000,'Style','infinite'); %'FaceLighting','gouraud',
light('Position',[-1 -1 1]*1000,'Style','infinite');
light('Position',[-.5 -.5 -11]*1000,'Style','infinite');
axis tight; axis equal;
view([-1 -1   0  ])
xlabel('X-AXIS'); 
ylabel('Y-AXIS'); 
zlabel('Z-AXIS');








