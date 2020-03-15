function [X_ba,P_ba,id_ba] = bundle_adjustment1(P,x,id,points_reconstruct,id_reconstruct,points_3d)
    % lamda*[x{i};1] = P{i}*[X;1]
    % [x{i};1] X (P{i}*[X;1]) = 0
    % 选出所有投影次数大于1的特征点
    all_id = cell2mat(id);
    id_tab = tabulate(all_id);
    id_ba = id_tab(id_tab(:,2) > 1)';
    id_num = length(id_ba);
    picture_num = length(x);
    call_loss(P,x,id,points_reconstruct,id_reconstruct,id_num,id_ba,picture_num)
    for step = 1:100
        % 计算所有X的梯度
        for n = 1:id_num
            X = points_reconstruct(:,find(id_reconstruct==id_ba(n)));
            uvw = [];
            pmat = [];
            for i = 1:picture_num
                if (any(id{i} == id_ba(n)))
                    uvw = [uvw;x{i}(:,find(id{i} == id_ba(n)))];
                    pmat = [pmat;P{i}];
                end
            end
            dLdX{n} = call_Xgrad(uvw,pmat,X);
        end
        % 计算所有P的梯度
        for n = 1:picture_num
            dLdP{n} = 0;
            pmat = P{n};
            for i = 1:length(id{n})
                uvw = x{n}(:,i);
                index = find(id_reconstruct==id{n}(i));
                if (index > 0)
                    X = points_reconstruct(:,index);                
                    dLdP{n} = dLdP{n} + call_Pgrad(uvw,pmat,X);
                end
            end
        end
        % 下降
        Xrate = sqrt(step)/1000000;
        Prate = sqrt(step);
        for n = 1:id_num
            index = find(id_reconstruct==id_ba(n));
            X = points_reconstruct(:,index);
            points_reconstruct(:,index) = X - Xrate * dLdX{n}';
        end
        for n = 1:picture_num
            P{n} = P{n} - Prate * dLdP{n}/(10^log10(abs(dLdP{n}(1,1))));
        end
        call_loss(P,x,id,points_reconstruct,id_reconstruct,id_num,id_ba,picture_num)
        filter_and_disp(points_reconstruct,id_reconstruct,points_3d);
    end
end

function dLdX = call_Xgrad(uvw,pmat,X)
        xx = pmat*[X;1];
        A = xx(find(mod(1:length(xx),3) == 0));
        B = xx(find(mod(1:length(xx),3) == 1));
        C = xx(find(mod(1:length(xx),3) == 2));
        BdA = B./A;
        CdA = C./A;
        D = uvw(find(mod(1:length(uvw),2) == 1));
        E = uvw(find(mod(1:length(uvw),2) == 0));
        % dL/dX = sum(2(BdA-D).*dBdA/dX) + sum(2(CdA-E).*dCdA/dX)
        % dBdA/dX = d(B/A)/dX = (dB/dX*A-B*dA/dX)/A^2
        % dB/dX = pmat(find(mod(1:length(xx),3) == 1),1:3)
        % dA/dX = pmat(find(mod(1:length(xx),3) == 0),1:3)
        dBdX = pmat(find(mod(1:length(xx),3) == 1),1:3);
        dCdX = pmat(find(mod(1:length(xx),3) == 2),1:3);
        dAdX = pmat(find(mod(1:length(xx),3) == 0),1:3);
        dLdX = sum(2*(BdA-D).*((dBdX.*A-dAdX.*B)./(A.^2)))...
             + sum(2*(CdA-E).*((dCdX.*A-dAdX.*C)./(A.^2)));
end

function dLdP = call_Pgrad(uvw,pmat,X)
        xx = pmat*[X;1];
        A = xx(find(mod(1:length(xx),3) == 0));
        B = xx(find(mod(1:length(xx),3) == 1));
        C = xx(find(mod(1:length(xx),3) == 2));
        BdA = B./A;
        CdA = C./A;
        D = uvw(find(mod(1:length(uvw),2) == 1));
        E = uvw(find(mod(1:length(uvw),2) == 0));
        % dL/dX = sum(2(BdA-D).*dBdA/dX) + sum(2(CdA-E).*dCdA/dX)
        % dBdA/dX = d(B/A)/dX = (dB/dX*A-B*dA/dX)/A^2
        % dB/dX = pmat(find(mod(1:length(xx),3) == 1),1:3)
        % dA/dX = pmat(find(mod(1:length(xx),3) == 0),1:3)
        dLdP1 = 2*(BdA-D).*([X' 1]./A);
        dLdP2 = 2*(CdA-E).*([X' 1]./A);
        dLdP3 = 2*(BdA-D).*((-B.*[X' 1])./(A.^2))+2*(CdA-E).*((-C.*[X' 1])./(A.^2));
        dLdP = [combine_col(dLdP1(:,1),dLdP2(:,1),dLdP3(:,1)) combine_col(dLdP1(:,2),dLdP2(:,2),dLdP3(:,2))...
                combine_col(dLdP1(:,3),dLdP2(:,3),dLdP3(:,3)) combine_col(dLdP1(:,4),dLdP2(:,4),dLdP3(:,4))];
end

function loss = call_loss(P,x,id,points_reconstruct,id_reconstruct,id_num,id_ba,picture_num)
    loss = 0;    
    for n = 1:id_num
        X = points_reconstruct(:,find(id_reconstruct==id_ba(n)));
        uvw = [];    
        pmat = [];
        for i = 1:picture_num
            if (any(id{i} == id_ba(n)))
                uvw = [uvw;x{i}(:,find(id{i} == id_ba(n)))];
                pmat = [pmat;P{i}];
            end
        end
        loss = loss + call_local_loss(uvw,pmat,X);
    end    
end

function loss = call_local_loss(uvw,P,X)
    xx = P*[X;1];
    A = xx(find(mod(1:length(xx),3) == 0));
    B = xx(find(mod(1:length(xx),3) == 1));
    C = xx(find(mod(1:length(xx),3) == 2));
    B = B./A;
    C = C./A;
    D = uvw(find(mod(1:length(uvw),2) == 1));
    E = uvw(find(mod(1:length(uvw),2) == 0));
    loss = [B-D;C-E];
    loss = sum(loss.^2);
end