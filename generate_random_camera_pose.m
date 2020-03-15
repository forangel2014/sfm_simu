function [R t] = generate_random_camera_pose(varargin)
    t = rand(3,1)*400-200;
    if (nargin == 1)
        center = cell2mat(varargin(1));
        while (in_pc(t))
            t = rand(3,1)*400-200;
        end
    else
        center = cell2mat(varargin(1));
        region = cell2mat(varargin(2));
        while (in_pc(t))
            t = region(:,1) + (region(:,2)-region(:,1)).*rand();
        end
    end
    z_target = center - t;
    z_target = z_target/norm(z_target);
    % z_target = R*[0;0;1]
    phi = acos(z_target(3));
    sin_theta = z_target(1)/sin(phi);
    cos_theta = -z_target(2)/sin(phi);
    R = [cos_theta -sin_theta*cos(phi) sin_theta*sin(phi);
        sin_theta cos_theta*cos(phi) -cos_theta*sin(phi);
        0 sin(phi) cos(phi)];
end