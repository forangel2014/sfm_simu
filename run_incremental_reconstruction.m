%% 增量重构
% 生成输入的重构序列
clearvars -except points_reconstruct_init id_reconstruct_init center K points_3d P x id
points_reconstruct = points_reconstruct_init;
id_reconstruct = id_reconstruct_init;
reg_num = 100;
for n = 1:reg_num
    [R_reg{n},t_reg{n}] = generate_random_camera_pose(center);
    % 计算真实E、F
    E_reg{n} = [cross(t_reg{n},R_reg{n}(:,1)) cross(t_reg{n},R_reg{n}(:,2)) cross(t_reg{n},R_reg{n}(:,3))];
    F_reg{n} = inv(K')*E_reg{n}*inv(K);
    [points_visable_reg{n},id_reg{n}] = select_points(t_reg{n},center,points_3d);
    points_project_reg{n} = project_points(K,R_reg{n},t_reg{n},points_visable_reg{n});
    flag(n) = 1;
end
% 重构过程
for reconstruct_num = 1:reg_num
    % 选择与已重构点云匹配数最大的一张
    for n = 1:reg_num
        intersect_point_num(n) = length(intersect(id_reconstruct,id_reg{n}));
    end
    overlap = sortrows([intersect_point_num; 1:reg_num]',1)';
    for n = reg_num:-1:1
        if (flag(overlap(2,n)) == 1)
            reconstruct_picture_id = overlap(2,n);
            break;
        end
    end
    % 生成匹配
    matched = generate_match(id_reconstruct,id_reg{reconstruct_picture_id},0);
    points_reg_2d = points_project_reg{reconstruct_picture_id}(:,matched(matched ~= 0));
    points_rec_3d = points_reconstruct(:,matched ~= 0);
    id_reg_2d = id_reg{reconstruct_picture_id}(matched(matched ~= 0));
    id_rec_3d = id_reconstruct(matched ~= 0);
    [C, R] = LinearPnP(points_rec_3d', points_reg_2d', K);
    t = -R*C;
    P{reconstruct_num+2} = K*[R t];
    x{reconstruct_num+2} = points_project_reg{reconstruct_picture_id};
    id{reconstruct_num+2} = id_reg{reconstruct_picture_id};
    % 三角化
    [points_reconstruct,id_reconstruct] = n_triangulation(P,x,id);
    % ba
    filter_and_disp(points_reconstruct,id_reconstruct,points_3d);
    % deltaX = inv(J'*J)*J'*(b-f(X))
    [points_reconstruct,P_reconstruct,id_reconstruct] = bundle_adjustment1(P,x,id,points_reconstruct,id_reconstruct,points_3d);
    flag(reconstruct_picture_id) = 0;
    [points_reconstruct,id_reconstruct] = filter_and_disp(points_reconstruct,id_reconstruct,points_3d);
end