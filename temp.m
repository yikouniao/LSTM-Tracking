clear; clc;

%fname='../train/MOT17-02-FRCNN/img1/000275.jpg'
%draw_bb(fname,457,458,30,104);

% layer = lstmLayer('InputSize', 2, 'OutputSize', 8, 'OutputMode', 'last');
% 
% options = trainingOptions('sgdm', 'ExucutionEnvironment', 'gpu',...
%     'InitialLearnRate', 0.002, 'LearnRateSchedule', 'piecewise',...
%     'LearnRateDropFactor', 0.1, 'LearnRatePeriod', 10,...
%     'MaxEpochs', 50, 'MiniBatchSize', '64', 'Momentum', 0.9,...
%     'Shuffle', 'every-epoch', 'Verbose', 1, 'VerboseFrequency', 1,...
%     'Plots', 'training-progress0');

% clear; clc;
% 
% fpath = '../MOT17/train/';
% foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
%     'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];
% resolution = [[1920, 1080]; [1920, 1080]; [640, 480]; [1920, 1080];...
%     [1920, 1080]; [1920, 1080]; [1920, 1080]];
% fps = [30; 30; 14; 30; 30; 30; 25];
% split_factor = 0.7;
% 
% p_x_train = [];
% p_y_train = [];
% p_x_test = [];
% p_y_test = [];
% 
% seqnum = size(foldername, 1);
% for i = 1:seqnum
%     gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
%     dets = load(gt_fname);
% 
%     % uses valid pedestrian data only to train
%     dets = dets(dets(:, 7)==1 & dets(:, 8)==1 & dets(:, 9)>0.8, :);
%     dets = dets(1, :);
%     draw_bb([fpath foldername(i, :) '/img1/000088.jpg'], dets(3), dets(4), dets(5), dets(6))
% end

a = zeros(20, 9);
for i = 1:size(a, 1) - 8
    i
end