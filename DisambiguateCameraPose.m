function [C,R,X0] = DisambiguateCameraPose(Cset,Rset,Xset,scale,center)
    for i = 1:4
        num(i) = sum((Xset{i}(:,3)*scale > 40).*(Xset{i}(:,3)*scale < 160));
        t(:,i) = Cset{i}*scale;
        dir(i) = (center-t(i))'*(Rset{i}*[0;0;1]);
    end
    [~,i] = max(num+dir);
    C = Cset{i};
    R = Rset{i};
    X0 = Xset{i};
end