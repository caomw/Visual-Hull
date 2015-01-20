
cls
path = dir('.\results');

len = size(path, 1);

counter = 0;
figure;
for i = 3 : len,
    disp(['processing', num2str(i-2), 'th frame ... ...']);
    img = imread(['.\results\', path(i).name]);
    imagesc(img);    colorbar;
    drawnow;
    pause(0.001);
end