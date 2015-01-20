
cls
path = dir('.\results');

len = size(path, 1);

counter = 0;
for i = 3 : len,
    img = imread(['.\results\', path(i).name]);
    disp(sum(sum(img)));
end