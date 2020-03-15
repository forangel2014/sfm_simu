function matched = generate_match(id1,id2,p)
    id = intersect(id1,id2);
    for i = 1:length(id1)
        temp = rand();
        if (any(id == id1(i)))
            if (temp < p)
                matched(i) = unidrnd(length(id2));
            else
                matched(i) = find(id2==id1(i));
            end
        else
            if (temp < p)
                matched(i) = unidrnd(length(id2));
            else
                matched(i) = 0;
            end
        end
    end
end