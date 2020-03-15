function [X_ba,P_ba,id_ba] = bundle_adjustment(P,x,id,points_reconstruct,id_reconstruct)
    % lamda*[x{i};1] = P{i}*[X;1]
    % [x{i};1] X (P{i}*[X;1]) = 0
    % 选出所有投影次数大于1的特征点
    all_id = cell2mat(id);
    id_tab = tabulate(all_id);
    id_ba = id_tab(id_tab(:,2) > 1)';
    id_num = length(id_ba);
    picture_num = length(x);
    for n = 1:id_num
        uvw = [];
        pmat = [];
        for i = 1:picture_num
            if (any(id{i} == id_ba(n)))
                uvw = [uvw;x{i}(:,find(id{i} == id_ba(n)))];
                pmat = [pmat;P{i}];
            end
        end
        % uvw = pmat*[X;1]
        X_init = points_reconstruct(:,find(id_reconstruct==id_ba(n)));
        call_loss(uvw,pmat,X_init)
        [X_ba(1:3,n) P_ba{n} flag(n)] = GD1(uvw,pmat,X_init);
    end
    index = find(flag);
    X_ba = X_ba(:,index);
    P_ba = P_ba{index};
    id_ba = id_ba(index);
end