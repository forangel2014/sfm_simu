function points_project = project_points(K,R,t,points_3d)
    points_project = K*[R t]*[points_3d;ones(1,size(points_3d,2))];
    points_project = points_project(1:2,:)./points_project(3,:) + rand(2,1)*2-1;
end