clear all
close all
clc

% Import camcal files -- formatted in MayaCam format, as exported fro
% XMALab -- and export in compatct format for BJTML
% Will Burton

data_root="/home/will/Desktop/research/BJTML/BJTML/example_studies/DU_VAL1/";
camcal_path1=data_root+"/DATA/Subject170/Camcal/Camcal0.txt";
camcal_path2=data_root+"/DATA/Subject170/Camcal/Camcal1.txt";
path_out=    data_root+"/DATA/Subject170/camcal_BJTML.txt";

camcal1=get_camcal(camcal_path1);
camcal2=get_camcal(camcal_path2);

R01=camcal2.R*camcal1.R';
T01=camcal2.T-camcal2.R*camcal1.R'*camcal1.T;

R10=camcal1.R*camcal2.R';
T10=camcal1.T-camcal1.R*camcal2.R'*camcal2.T;

% Get B origin in cam A cosys
%O2_1=R01'*([0;0;0]-T01);

% Get B basis vectors in cam A coys
%B2_2=eye(3);
%B2_1=R01'*B2_2;

R10'*R10
det(R10)

DU_INTCALIB_BIPLANE=[
         camcal1.fx;
         camcal1.fy;
         camcal1.cx;
         camcal1.cy;
         camcal1.IM(1);
         camcal1.IM(2);
         ...
         camcal2.fx;
         camcal2.fy;
         camcal2.cx;
         camcal2.cy;
         camcal2.IM(1);
         camcal2.IM(2);
         ...
         R10(:,1);
         R10(:,2);
         R10(:,3);
         T10;
        ];

my_table=table(DU_INTCALIB_BIPLANE);

%writematrix(vec_out, path_out);
writetable(my_table, path_out); % Must write with header for BJTML compatibility

% NOTE: IM FORMAT IS [WIDTH,HEIGHT]
% Verify by looking at Mayacam export function (version 2) in XMALAB source
% code -- i.e., go to:
% https://bitbucket.org/xromm/xmalab/src/master/src/core/Camera.cpp
% CTRL+F for "saveMayaCamVersion2"

my_vec=[0;0;1300];
my_vec_B=R01*my_vec+T01;
my_vec_B2=camcal2.R*(camcal1.R'*(my_vec-camcal1.T))+camcal2.T;
my_vec_B3=R10'*(my_vec-T10);















% DU_INTCALIB_BIPLANE=[
%          camcal1.fx;
%          camcal1.fy;
%          camcal1.cx;
%          camcal1.cy;
%          camcal1.IM(1);
%          camcal1.IM(2);
%          ...
%          camcal2.fx;
%          camcal2.fy;
%          camcal2.cx;
%          camcal2.cy;
%          camcal2.IM(1);
%          camcal2.IM(2);
%          ...
%          R01(:,1);
%          R01(:,2);
%          R01(:,3);
%          T01;
%         ];


