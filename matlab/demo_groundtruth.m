
%% initialization
initial
    
%% main
for mainloop_i= 10 : 10,
    %% read images
    disp(sprintf('Processing %d image...', mainloop_i));
    img1 = imread(sprintf('G:/experiments/Fview1/view0/tview0/img%06d.jpg', mainloop_i));
    img2 = imread(sprintf('G:/experiments/Fview1/view1/tview1/img%06d.jpg', mainloop_i));
    img3 = imread(sprintf('G:/experiments/Fview1/view2/tview2/img%06d.jpg', mainloop_i));
    img4 = imread(sprintf('G:/experiments/Fview1/view3/tview3/img%06d.jpg', mainloop_i));
    
    %% visual hull
    space_cubes = feval(k, ...
    zeros(thready*blocky, threadx*blockx, threadz),...
    img1, img2, img3, img4, ...
    mT, ...
    P1, P2, P3, P4, ...
    ratiox, ratioy, ratioz, ...
    640, 480);

    cpu_space_cubes = gather(space_cubes);
  
    %% ground projection
%     img_ground = sum(cpu_space_cubes, 3);
%     hold on
%     imagesc(img_ground);


    hold on;
    
    space_size = size(cpu_space_cubes);

    temp = reshape(cpu_space_cubes, space_size(1), space_size(2)*space_size(3));

    [temp_size_x, temp_size_y] = find(temp==1);

    point_cloud_size = size(temp_size_x, 1);

    x = temp_size_x;
    y = mod(temp_size_y, space_size(3));
    z = ceil(temp_size_y/space_size(3));

    scatter3(x,y,z);

    drawnow;
    pause(1);
end