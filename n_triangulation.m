function [X_tri,id_tri] = n_triangulation(P,x,id)
%% LinearTriangulation
% Find 3D positions of the point correspondences using the relative
% position of one camera from another
% Inputs:
%     K  - size (3 x 3) camera intrsinc parameter for both cameras
%     C1 - size (3 x 1) translation of the first camera pose
%     R1 - size (3 x 3) rotation of the first camera pose
%     C2 - size (3 x 1) translation of the second camera
%     R2 - size (3 x 3) rotation of the second camera pose
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Outputs: 
%     X - size (N x 3) matrix whos rows represent the 3D triangulated
%       points
    all_id = cell2mat(id);
    id_tab = tabulate(all_id);
    id_tri = id_tab(id_tab(:,2) > 1)';
    id_num = length(id_tri);
    picture_num = length(x);
    for n = 1:id_num
        A = [];
        for i = 1:picture_num
            if (any(id{i} == id_tri(n)))
                x1 = x{i}(:,find(id{i} == id_tri(n)));
                x_mat = Vec2Skew([x1;1]);
                A = [A;x_mat*P{i}];  
            end
        end
        [~,~,V] = svd(A);
        X_b = V(1:3,end)/V(end,end);
        X_tri(:,n) = X_b;
    end
end
