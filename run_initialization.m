clc, clear all, close all;
points_num = 10000;
points_3d = [100 0 0; 0 100 0; 0 0 100]*rand(3,points_num) + [-50;-50;50];
center = [0;0;100];
K = [568.996140852000,0,643.210559410000;
    0,568.988362396000,477.982801038000;
    0,0,1];

%% 初始化（虚拟了SIFT）
% 随机生成相机姿态 视线方向对象点云中心
R1 = eye(3); t1 = zeros(3,1);
[R2,t2] = generate_random_camera_pose(center,[-50,50;-50,50;-5,5]);
% 计算真实E、F
E_real = [cross(t2,R2(:,1)) cross(t2,R2(:,2)) cross(t2,R2(:,3))];
F_real = inv(K')*E_real*inv(K);
% 计算能够投影的特征点
[points_visable1,id1] = select_points(t1,center,points_3d);
[points_visable2,id2] = select_points(t2,center,points_3d);
% 投影特征点
points_project1 = project_points(K,R1,t1,points_visable1);
points_project2 = project_points(K,R2,t2,points_visable2);
% 生成匹配（随机产生错误匹配）
matched = generate_match(id1,id2,0.1);
id_gen1 = id1(:,matched ~= 0);
id_gen2 = id2(:,matched(matched ~= 0));
% ransac
[matched,max_inliers,F] = my_ransac(matched,points_project1',points_project2');
points_matched1 = points_project1(:,matched ~= 0);
points_matched2 = points_project2(:,matched(matched ~= 0));
id_matched1 = id1(:,matched ~= 0);
id_matched2 = id2(:,matched(matched ~= 0));
% 根据局内点重新计算最小mse下的F
F = EstimateFundamentalMatrix(points_matched1', points_matched2');
F = F*norm(F_real);
E = EssentialMatrixFromFundamentalMatrix(F,K);
% 重构相机姿态
[Cset,Rset] = ExtractCameraPose(E);
X{1} = LinearTriangulation(K, zeros(3,1), eye(3), Cset{1}, Rset{1}, points_matched1', points_matched2');
X{2} = LinearTriangulation(K, zeros(3,1), eye(3), Cset{2}, Rset{2}, points_matched1', points_matched2');
X{3} = LinearTriangulation(K, zeros(3,1), eye(3), Cset{3}, Rset{3}, points_matched1', points_matched2');
X{4} = LinearTriangulation(K, zeros(3,1), eye(3), Cset{4}, Rset{4}, points_matched1', points_matched2');
[C,R,X0] = DisambiguateCameraPose(Cset,Rset,X,norm(t2),center);
t = -R*C*norm(t2);
% 在原尺度上恢复点云
points_reconstruct = Rt_triangulation(K,R1,t1,R,t,points_matched1, points_matched2);
points_real = points_3d(:,id_matched1);
id_reconstruct = id_matched1;
P{1} = K*[R1 t1];
P{2} = K*[R2 t2];
x{1} = points_project1;
x{2} = points_project2;
id{1} = id1;
id{2} = id2;
[points_reconstruct_init,id_reconstruct_init] = filter_and_disp(points_reconstruct,id_reconstruct,points_3d);