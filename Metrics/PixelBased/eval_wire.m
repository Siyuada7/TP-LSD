function eval_wire(imagesPath, dSetPath, resultPath, method, outFile, lineThresh, output_size, suffix)

maxDist = 0.01;
divide_eps = 1e-15;
nLineThresh = size(lineThresh, 2);
sumprecisions = zeros(nLineThresh, 1);
sumrecalls = zeros(nLineThresh, 1);
nsamples = zeros(nLineThresh, 1);

listing = dir(imagesPath);
numResults = size(listing, 1);

%for index=1:numResults
for index=1:numResults
    filename = listing(index).name;
    if length(filename) == 1 || length(filename) == 2
        continue;
    end
    filename = filename(1:end-4);
    disp([num2str(index), ' == ', filename])
    gtname = [dSetPath, filename, '_line.mat'];
    imgname = [imagesPath, filename, suffix];
    
    I = imread(imgname);
    height = size(I,1);
    width = size(I,2);
    
    %% convert GT lines to binary map
    gtlines = load(gtname);
    gtlines = double(gtlines.lines);
    
    ne = size(gtlines,1);
    edgemap0 = zeros(height, width);
    for k = 1:ne
        x1 = gtlines(k,1);
        x2 = gtlines(k,3);
        y1 = gtlines(k,2);
        y2 = gtlines(k,4);
        
        vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
        cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
        for j = 1:size(cur_edge,1)
            yy = round(cur_edge(j,1));
            xx = round(cur_edge(j,2));
            if yy <= 0
                yy = 1;
            end
            if xx <= 0
                xx = 1;
            end
            if yy > height
                    yy = height;
                end
                if xx > width
                    xx = width;
                end
            edgemap0(yy,xx) = 1;
        end
    end
    
    parfor m=1:nLineThresh
        %for m=1:nLineThresh
        if strcmp(method,'lcnn_post') || strcmp(method,'lcnn')
            name = num2str(index-3,'%06d');
            resultname = [resultPath, '/', num2str(lineThresh(m)), '/', name, '.mat'];
        else
            resultname = [resultPath, '/', num2str(lineThresh(m)), '/', filename, '.mat'];
        end
        resultlines = load(resultname);
        resultlines = double(resultlines.lines);
        ne = size(resultlines,1);
        disp([' ', num2str(lineThresh(m)), ' #lines: ' num2str(ne)]);
        
        edgemap1 = zeros(height, width);
        for k = 1:ne
            if strcmp(method,'tplsd')
                x1 = resultlines(k,1) * width / output_size; 
                y1 = resultlines(k,2)* height / output_size;
                x2 = resultlines(k,3)* width / output_size;
                y2 = resultlines(k,4)* height / output_size;
            elseif strcmp(method,'lsd')
                x1 = resultlines(k,2); 
                y1 = resultlines(k,1);
                x2 = resultlines(k,4);
                y2 = resultlines(k,3);
            else
                x1 = resultlines(k,1);
                y1 = resultlines(k,2);
                x2 = resultlines(k,3);
                y2 = resultlines(k,4);
            end
            
            vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
            cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
            for j = 1:size(cur_edge,1)
                yy = round(cur_edge(j,1));
                xx = round(cur_edge(j,2));
                if yy <= 0
                    yy = 1;
                end
                if xx <= 0
                    xx = 1;
                end
                if yy > height
                    yy = height;
                end
                if xx > width
                    xx = width;
                end
                edgemap1(yy,xx) = 1;
            end
        end
        
        [matchE1,matchG1] = correspondPixels(edgemap1,edgemap0,maxDist);
        matchE = double(matchE1>0);
        matchG = double(matchG1>0);
        
        % compute recall (summed over each gt image)
        cntR = sum(matchG(:)); sumR = sum(edgemap0(:));
        recall = cntR / (sumR + divide_eps);
        % compute precision (edges can match any gt image)
        cntP = sum(matchE(:)); sumP = sum(edgemap1(:)); precision = cntP / (sumP + divide_eps);
        disp(['===== filename = ' filename ', precision = ' num2str(precision) ', recall = ' num2str(recall)]);
        sumprecisions(m, 1) = sumprecisions(m, 1) + precision;
        sumrecalls(m, 1) = sumrecalls(m, 1) + recall;
        nsamples(m, 1) = nsamples(m, 1) + 1;
    end
end
disp([sumprecisions; sumrecalls])
save(outFile, 'sumprecisions', 'sumrecalls', 'nsamples');

end

