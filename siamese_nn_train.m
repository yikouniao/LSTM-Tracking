clear; clc;

fpath = '../MOT17/train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];
split_factor = 0.7;

seqnum = size(foldername, 1);
for i = 1:seqnum
    pos_cnt = 0;
    neg_cnt = 0;
    id_record = [];
    id_cnt_record = [];
    frame_record = [];
    frame_cnt_record = [];
    gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
    dets = load(gt_fname);

    % uses valid pedestrian data only to train
    dets = dets(dets(:, 7)==1 & dets(:, 8)==1 & dets(:, 9)>0.8, :);
    dets = dets(:, 1:6);
    
	% forms the positive samples within identities
    max_id = max(dets(:, 2));
    for j = 1:max_id % for each id
        if sum(dets(:, 2)==j) < 2
            continue;
        end
        
        id_record = [id_record; j];
        id_dets = dets(dets(:, 2)==j, :);
        id_cnt_record = [id_cnt_record; size(id_dets, 1)];
    end

    for j = 1:size(id_cnt_record, 1)
        n = id_cnt_record(j, 1);        
        pos_cnt = pos_cnt + n * (n - 1) / 2;
    end
    
    x1_pos = zeros(pos_cnt, 6);
    x2_pos = zeros(pos_cnt, 6);
    y_pos = ones(pos_cnt, 1);
    pos_iter = 1;
    
    for j = 1:size(id_record, 1)
        id = id_record(j, 1);
        n = id_cnt_record(j, 1);
        id_dets = dets(dets(:, 2)==id, :);
        for k = 1:n-1
            for l = k+1:n
                x1_pos(pos_iter, :) = id_dets(k, :);
                x2_pos(pos_iter, :) = id_dets(l, :);
                pos_iter = pos_iter + 1;
            end
        end
    end
    
    split = fix(pos_cnt * split_factor);
    x1_pos_train = x1_pos(1:split, :);
    x1_pos_test = x1_pos(split + 1:pos_cnt, :);
    x2_pos_train = x2_pos(1:split, :);
    x2_pos_test = x2_pos(split + 1:pos_cnt, :);
    y_pos_train = y_pos(1:split, :);
    y_pos_test = y_pos(split + 1:pos_cnt, :);
    
    % forms the negative samples within frames
    max_frame = max(dets(:, 1));
    for j = 1:max_frame % for each frame
        if sum(dets(:, 1)==j) < 2
            continue;
        end
        
        frame_record = [frame_record; j];
        frame_dets = dets(dets(:, 1)==j, :);
        frame_cnt_record = [frame_cnt_record; size(frame_dets, 1)];
    end

    for j = 1:size(frame_cnt_record, 1)
        n = frame_cnt_record(j, 1);        
        neg_cnt = neg_cnt + n * (n - 1) / 2;
    end
    
    x1_neg = zeros(neg_cnt, 6);
    x2_neg = zeros(neg_cnt, 6);
    y_neg = zeros(neg_cnt, 1);
    neg_iter = 1;
    
    for j = 1:size(frame_record, 1)
        frame = frame_record(j, 1);
        n = frame_cnt_record(j, 1);
        frame_dets = dets(dets(:, 1)==frame, :);
        for k = 1:n-1
            for l = k+1:n
                x1_neg(neg_iter, :) = frame_dets(k, :);
                x2_neg(neg_iter, :) = frame_dets(l, :);
                neg_iter = neg_iter + 1;
            end
        end
    end
    
    split = fix(neg_cnt * split_factor);
    x1_neg_train = x1_neg(1:split, :);
    x1_neg_test = x1_neg(split + 1:neg_cnt, :);
    x2_neg_train = x2_neg(1:split, :);
    x2_neg_test = x2_neg(split + 1:neg_cnt, :);
    y_neg_train = y_neg(1:split, :);
    y_neg_test = y_neg(split + 1:neg_cnt, :);
    
    % splice negative and postive samples
    x1_train_bb = [x1_pos_train; x1_neg_train];
    x1_test_bb = [x1_pos_test; x1_neg_test];
    x2_train_bb = [x2_pos_train; x2_neg_train];
    x2_test_bb = [x2_pos_test; x2_neg_test];
    y_train_bb = [y_pos_train; y_neg_train];
    y_test_bb = [y_pos_test; y_neg_test];
    
    save([fpath foldername(i, :) '/a/x1_train_bb.txt'], 'x1_train_bb');
    save([fpath foldername(i, :) '/a/x1_test_bb.txt'], 'x1_test_bb');
    save([fpath foldername(i, :) '/a/x2_train_bb.txt'], 'x2_train_bb');
    save([fpath foldername(i, :) '/a/x2_test_bb.txt'], 'x2_test_bb');
    save([fpath foldername(i, :) '/a/y_train_bb.txt'], 'y_train_bb');
    save([fpath foldername(i, :) '/a/y_test_bb.txt'], 'y_test_bb');
end