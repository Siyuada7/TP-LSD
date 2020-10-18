lineThresh = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8];
method = 'tplsd';

%% Wire
dSetPath = 'path for wireframe valid labels';
imagesPath = 'path for wireframe valid labels images';
suffix = '.jpg';

output_size=320;
resultPath = 'path for line detection result /TP-LSD/wire/mat/lmbd0.5/';
outFile = ['path for result /TP-LSD/wire/mat/lmbd0.5/result.mat'];
eval_wire(imagesPath, dSetPath, resultPath, method, outFile, lineThresh, output_size, suffix);

%% York
dSetPath = 'path for YorkUrban valid labels';
imagesPath = 'path for YorkUrban valid labels images';
suffix = '.png';

resultPath = 'path for line detection result /TP-LSD/york/mat/lmbd0.5/';
outFile = ['path for result /TP-LSD/york/mat/lmbd0.5/result.mat'];
eval_wire(imagesPath, dSetPath, resultPath, method, outFile, lineThresh, output_size, suffix);

