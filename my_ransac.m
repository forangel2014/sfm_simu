function [best_matched,max_inliers,best_F] = my_ransac(matched,loc1,loc2)  
    matched = [1:length(matched);matched];
    id_matched = matched(:,matched(2,:) ~= 0);
    % id_matched: [...ai ai+1...
    %              ...bi bi+1...]
    % loc1(ai) matched loc2(bi)
    matched_num = size(id_matched,2);
    best_F = zeros(3,3);
    best_matched = [];
    max_inliers = 0;
    Thres = 1e-4;
    for iteration = 1:20000
        this_matched = matched;
        order = randperm(matched_num);
        pix1 = loc1(id_matched(1,order(1:8)),:);
        pix2 = loc2(id_matched(2,order(1:8)),:);
        F = EstimateFundamentalMatrix(pix1, pix2);
        inliers = 0;
        se = 0;
        for i = 1:matched_num
            p1 = loc1(id_matched(1,i),:);
            p2 = loc2(id_matched(2,i),:);
            error = ([p2 1]*F*[p1 1]')^2;
            if (error <= Thres)
                inliers = inliers+1;
                se = se+error;
            else
                this_matched(:,id_matched(1,i)) = 0;
            end
        end
        if (inliers >= max_inliers)
            max_inliers = inliers;
            best_F = F;
            best_matched = this_matched(2,:);
        end
    end
end