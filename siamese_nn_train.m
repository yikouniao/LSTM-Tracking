clear; clc;

fpath = '../MOT17/train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];
split_factor = 0.7;

% x1 = [];
% x2 = [];
% y = [];
pos_cnt = 0;
neg_cnt = 0;
id_record = [];
id_cnt_record = [];

seqnum = size(foldername, 1);
for i = 1:seqnum
    gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
    dets = load(gt_fname);

    % uses valid pedestrian data only to train
    dets = dets(dets(:, 7)==1 & dets(:, 8)==1 & dets(:, 9)>0.8, :);
    dets = dets(:, 1:6);
    
	% forms the positive samples within identities
    max_id = max(dets(:, 2));
    for j = 1:max_id % for each id
        if sum(dets(:, 2)==j) < 1
            continue;
        end
        
        id_record = [id_record; j];
        id_dets = dets(dets(:, 2)==j, :);
        id_cnt_record = [id_cnt_record; size(id_dets, 1)];
    end
%         for k = 1:size(id_dets, 1)-1
%             for l = k+1:size(id_dets, 1)
%                 x1 = [x1; id_dets(k, :)];
%                 x2 = [x2; id_dets(l, :)];
%                 y = [y; 1];
%             end
%         end
    for j = 1:size(id_record, 1)
        id = id_record(j, 1);
        n = id_cnt_record(j, 1);
        
        pos_cnt = pos_cnt + n * (n - 1) / 2;
    end
    
    %draw_bb([fpath foldername(i, :) '/img1/000088.jpg'], dets(3), dets(4), dets(5), dets(6))
end