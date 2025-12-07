clear all
close all
clc

path1="/home/will/Desktop/research/BJTML/BJTML/example_studies/UF1/KR_right_7_fem.stl";
stl=stlread(path1);

path2="/home/will/Desktop/research/BJTML/BJTML/example_studies/MIXED/femur.stl";
stl2=stlread(path2);

[e,n]=reducepatch(stl.ConnectivityList, stl.Points, 5000);

stl_new=struct;
stl_new.Points=n;
stl_new.ConnectivityList=e;

stlnew=triangulation(e,n);
path_out="/home/will/Desktop/research/BJTML/BJTML/example_studies/UF1/KR_right_7_fem_REDUCED.stl";
% stlwrite(stlnew, path_out);

figure;
scatter3(stl2.Points(:,1),stl2.Points(:,2),stl2.Points(:,3));
axis equal;
xlabel("x");
ylabel("y");
zlabel("z");