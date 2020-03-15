function X = Rt_triangulation(K,R1,t1,R,t,points_matched1, points_matched2)
    n = size(points_matched1,2);
    X = zeros(n,3);
    for i = 1:n
        x1 = Vec2Skew([points_matched1(:,i);1]);
        x2 = Vec2Skew([points_matched2(:,i);1]);
        A = [x1*K*[R1 t1];x2*K*[R t]];
        [~,~,V] = svd(A);
        X_b = V(1:3,end)/V(end,end);
        X(i,:) = X_b.';
    end
    X = X';
end