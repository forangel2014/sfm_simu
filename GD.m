function [X_rec,flag] = GD(uvw,pmat,X_init)
    flag = 0;
    X(:,1) = X_init;
    reso1 = 1;
    reso2 = 1;
    for step = 1:100
        dLdX = call_grad(uvw,pmat,X(:,step));
        dLdX2 = call_grad2(uvw,pmat,X(:,step));
        Loss = call_loss(uvw,pmat,X(:,step));
        % Loss = call_loss(X-rate*dLdX')
        % dLdRate = dLd(X-rate*dLdX')*(-dLdX');
        rate1 = 0.01*log10(Loss)*sqrt(step)*reso1;
        Y = X(:,step) - rate1*dLdX'/(norm(dLdX)) ;%+ 1/(sqrt(step)) * (rand(3,1)*10-5);
        loss1 = call_loss(uvw,pmat,Y);
        rate2 = 0.01*log10(Loss)*sqrt(step)*reso1;
        Z = X(:,step) - rate2*inv(dLdX2)*dLdX'/(norm(inv(dLdX2)*dLdX')) ;%+ 1/(sqrt(step)) * (rand(3,1)*10-5);
        loss2 = call_loss(uvw,pmat,Z);
        if (loss1 < loss2)
            X(:,step+1) = Y;
            loss(step) = loss1;
            if (step > 1)
                if (loss(step) - loss(step-1) < 5)
                    reso1 = reso1+0.1;
                else
                    if(loss(step) > loss(step-1))
                        reso1 = reso1/2;
                    end
                end
            end
        else
            X(:,step+1) = Z;
            loss(step) = loss2;
            if (step > 1)
                if (loss(step) - loss(step-1) < 5)
                    reso2 = reso2+0.1;
                else
                    if(loss(step) > loss(step-1))
                        reso2 = reso2/2;
                    end
                end
            end
        end
        
        disp(loss(step));
        if (loss(step) < 5)
            flag = 1;
            X_rec = X(:,step);
            break;
        end
        if (step > 1)
            if (loss(step) > loss(step-1) && loss(step-1) < 30)
                flag = 1;
                X_rec = X(:,step-1);
                break;
            end
        end
    end
    if (any(loss < 20))
        flag = 1;
        [~,i] = min(loss);
        X_rec = X(:,i);
    else
        flag = 0;
        X_rec = [0;0;0];
    end
end

function dLdX = call_grad(uvw,pmat,X)
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
function dLdX2 = call_grad2(uvw,pmat,X)
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
        dBdX1 = pmat(find(mod(1:length(xx),3) == 1),1);
        dCdX1 = pmat(find(mod(1:length(xx),3) == 2),1);
        dAdX1 = pmat(find(mod(1:length(xx),3) == 0),1);
        dBdX2 = pmat(find(mod(1:length(xx),3) == 1),2);
        dCdX2 = pmat(find(mod(1:length(xx),3) == 2),2);
        dAdX2 = pmat(find(mod(1:length(xx),3) == 0),2);
        dBdX3 = pmat(find(mod(1:length(xx),3) == 1),3);
        dCdX3 = pmat(find(mod(1:length(xx),3) == 2),3);
        dAdX3 = pmat(find(mod(1:length(xx),3) == 0),3);
        dLdX2 = [(sum(2*((dBdX1.*A-dAdX1.*B)./(A.^2)).*((dBdX.*A-dAdX.*B)./(A.^2))...
                + 2*(BdA-D).*(((dBdX.*dAdX1-dAdX.*dBdX1).*A.^2-(dBdX.*A-dAdX.*B)*2.*A.*dAdX1)./(A.^4)))...
                +sum(2*((dCdX1.*A-dAdX1.*C)./(A.^2)).*((dCdX.*A-dAdX.*C)./(A.^2))...
                + 2*(CdA-E).*(((dCdX.*dAdX1-dAdX.*dCdX1).*A.^2-(dCdX.*A-dAdX.*C)*2.*A.*dAdX1)./(A.^4))));
                
                (sum(2*((dBdX2.*A-dAdX2.*B)./(A.^2)).*((dBdX.*A-dAdX.*B)./(A.^2))...
                + 2*(BdA-D).*(((dBdX.*dAdX2-dAdX.*dBdX2).*A.^2-(dBdX.*A-dAdX.*B)*2.*A.*dAdX2)./(A.^4)))...
                +sum(2*((dCdX2.*A-dAdX2.*C)./(A.^2)).*((dCdX.*A-dAdX.*C)./(A.^2))...
                + 2*(CdA-E).*(((dCdX.*dAdX2-dAdX.*dCdX2).*A.^2-(dCdX.*A-dAdX.*C)*2.*A.*dAdX2)./(A.^4))));
                
                (sum(2*((dBdX3.*A-dAdX3.*B)./(A.^2)).*((dBdX.*A-dAdX.*B)./(A.^2))...
                + 2*(BdA-D).*(((dBdX.*dAdX3-dAdX.*dBdX3).*A.^2-(dBdX.*A-dAdX.*B)*2.*A.*dAdX3)./(A.^4)))...
                +sum(2*((dCdX3.*A-dAdX3.*C)./(A.^2)).*((dCdX.*A-dAdX.*C)./(A.^2))...
                + 2*(CdA-E).*(((dCdX.*dAdX3-dAdX.*dCdX3).*A.^2-(dCdX.*A-dAdX.*C)*2.*A.*dAdX3)./(A.^4))))]; 
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