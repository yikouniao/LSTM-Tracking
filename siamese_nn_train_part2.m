clear;% clc;

in_fpath = '../MOT17/train/';
out_fpath = '/media/lym/Elements/MOT17/train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
              'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN';
              'MOT17-13-FRCNN'];
width = 52;
height = 170;
channel = 3;

seqnum = size(foldername, 1);               
for i = 1:seqnum
    for j = 1:100
        x1_bb_fname = [in_fpath foldername(i, :) '/a/x1_bb_splice_' num2str(j) '.mat'];
        x1_bb_splice = load(x1_bb_fname);
        x1_bb_splice = x1_bb_splice.x1_bb_splice;
        bb_num = size(x1_bb_splice, 1);
        x1_bb_date = zeros(bb_num, height, width, channel, 'gpuArray');
        for k = 1:bb_num
            img = imread([in_fpath foldername(i, :) '/img1/' num2fname(x1_bb_splice(k, 1))]);
            img = imcrop(img, x1_bb_splice(k, 3:6));
            img = gpuArray(im2single(img));
            img = imresize(img, [height, width]);
            x1_bb_date(1, :, :, :) = img;
            save([out_fpath foldername(i, :) '/a/x1_bb_date_' num2str(j) '.mat'], 'x1_bb_date');
        end
    end
end