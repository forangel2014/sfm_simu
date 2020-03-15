function flag = in_pc(t)
    n = size(t,2);
    for i = 1:n
        if (t(1,i) >= -50 && t(1,i) <= 50 && t(2,i) >= -50 && t(2,i) <= 50 && t(3,i) >= 50 && t(3,i) <= 150)
            flag(i) = 1;
        else
            flag(i) = 0;
        end
    end
end