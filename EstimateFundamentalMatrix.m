function F = EstimateFundamentalMatrix(x1, x2)
%% EstimateFundamentalMatrix
% Estimate the fundamental matrix from two image point correspondences 
% Inputs:
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Output:
%    F - size (3 x 3) fundamental matrix with rank 2

N = size(x1,1);
A = [];
for i = 1:N
    u1 = x1(i,1);
    v1 = x1(i,2);
    u2 = x2(i,1);
    v2 = x2(i,2);
    A(i,1:9) = [u1*u2 u1*v2 u1 v1*u2 v1*v2 v1 u2 v2 1];
end
[~,~,V] = svd(A);
f = V(:,end);
F = [f(1) f(2) f(3); f(4) f(5) f(6); f(7) f(8) f(9)];
[U,D,V] = svd(F);
D(3,3) = 0;
F = U*D*V';
F = F/norm(F);
F = F';
end