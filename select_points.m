function [points_visable id] = select_points(t,center,points_3d)
    dir = center - t;
    points = points_3d - t;
    dis = dir'*points;
    id = [1:length(dis);dis];
    result = sortrows(id',2)';
    id = result(1,1:1000+round(rand()*400)-200);
    points_visable = points_3d(1:3,id);
end