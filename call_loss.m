function loss = call_loss(x,P,X)
    xx = P*[X;1];
    A = xx(find(mod(1:length(xx),3) == 0));
    B = xx(find(mod(1:length(xx),3) == 1));
    C = xx(find(mod(1:length(xx),3) == 2));
    B = B./A;
    C = C./A;
    D = x(find(mod(1:length(x),2) == 1));
    E = x(find(mod(1:length(x),2) == 0));
    loss = sum((B-D).^2)+sum((C-E).^2);
end