function [points_reconstruct,id_reconstruct] = filter_and_disp(points_reconstruct,id_reconstruct,points_3d)
    n = find(~in_pc(points_reconstruct));
    points_reconstruct(:,n) = [];
    id_reconstruct(:,n) = [];
    mean_reconstruct_error = mean(sum((points_3d(:,id_reconstruct)-points_reconstruct).^2,1))
end