function [X_rec,P_rec,flag] = GD1(uvw,pmat,X_init)
    flag = 0;
    X{1} = X_init;
    P{1} = pmat;
    reso1 = 1;
    reso2 = 1;
    for step = 1:100
        [dLdP,dLdX] = call_grad(uvw,P{step},X{step});
        loss(step) = call_loss(uvw,P{step},X{step});
        % Loss = call_loss(X-rate*dLdX')
        % dLdRate = dLd(X-rate*dLdX')*(-dLdX');
        rate = 0.01*log(loss(step))*(step^(1/3))*reso1;
        X{step+1} = X{step} - rate*dLdX'/(norm(dLdX)); %+ 1/(sqrt(step)) * (rand(3,1)*10-5);
        %X{step+1} = X{step};
        P{step+1} = P{step} - rate/100*dLdP/(norm(dLdP));
        
        disp(loss(step));
        if (loss(step) < 2)
            flag = 1;
            X_rec = X{step};
            P_rec = P{step};
            break;
        end
        if (step > 1)
            if (loss(step) > loss(step-1) && loss(step-1) < 30)
                flag = 1;
                X_rec = X{step-1};
                P_rec = P{step-1};
                break;
            end
        end
    end
    if (any(loss < 20))
        flag = 1;
        [~,i] = min(loss);
        X_rec = X{i};
        P_rec = X{i};
    else
        flag = 0;
        X_rec = [0;0;0];
        P_rec = pmat;
    end
end

function [dLdP,dLdX] = call_grad(uvw,pmat,X)
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
        dLdP1 = 2*(BdA-D).*([X' 1]./A);
        dLdP2 = 2*(CdA-E).*([X' 1]./A);
        dLdP3 = 2*(BdA-D).*((-B.*[X' 1])./(A.^2))+2*(CdA-E).*((-C.*[X' 1])./(A.^2));
        dLdP = [combine_col(dLdP1(:,1),dLdP2(:,1),dLdP3(:,1)) combine_col(dLdP1(:,2),dLdP2(:,2),dLdP3(:,2))...
                combine_col(dLdP1(:,3),dLdP2(:,3),dLdP3(:,3)) combine_col(dLdP1(:,4),dLdP2(:,4),dLdP3(:,4))];
end

function loss = call_loss(uvw,P,X)
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