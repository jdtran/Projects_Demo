% Bayes' Classifier
set = 40;
trial = 2;
m1 = []; % hand class
m2 = []; % trace class
m3 = []; % background class

% Create training classes by masking the hand, trace, and background
% Use every fifth frame of the training data A
for i = 5:5:set
    data_name = sprintf('frameA%i.png', i);
    mask_hand = sprintf('frameA%i_hand.jpg', i);
    mask_trace = sprintf('frameA%i_trace.jpg', i);
    
    img = imread(data_name);
    % Gaussian filter with sigma of 0.8 and converting to YCbCr
    img = imgaussfilt(img, 0.8);
    img = rgb2ycbcr(img);
    img = double(img);
    % Flags to make sure only the availble masks are read
    mh = false;
    mt = false;
    % Check if the mask for the hand exists
    if exist(mask_hand, 'file') == 2
        mask_hand = rgb2gray(imread(mask_hand));
        mask_hand = imbinarize(mask_hand);
        mh = true;
    end
    % Check if the mask for the trace exists
    if exist(mask_trace, 'file') == 2
        mask_trace = rgb2gray(imread(mask_trace));
        mask_trace = imbinarize(mask_trace);
        mt = true;
    end
    % Return Hand, Trace, and Background classes as an Nx3 matrix
    [hand, trace, bg] = get_classes(img, mask_hand, mask_trace, mh, mt);
    m1 = [m1; hand];
    m2 = [m2; trace];
    m3 = [m3; bg];
end

frame = 41;
in_name = sprintf('testE.png', frame);
out_name = sprintf('testE_YCbCr.png', frame);
testImg = imread(in_name);

testImg = imgaussfilt(testImg, .8);
testImg = imresize(testImg, .25);    
testImg = rgb2ycbcr(testImg);
figure, imshow(testImg);
testImg = double(testImg);

[output] = bayes(testImg, m1, m2, m3);
output(output==1) = 0;
output(output==2) = 126;
output(output==3) = 255;
output = uint8(output);
figure, imshow(output);
imwrite(output, out_name);

function [m1, m2, m3] = get_classes(img, mask_hand, mask_trace, mh, mt)
    % m1 is strawberry class, m2 is non-strawberry class
    % Convert mask back into binary, it got scaled to 0-255 on imread
    m1 = [];
    m2 = [];
    m3 = [];
    
    % Segment the training images with the mask and making sure masks
    % exists because not all data have a hand or trace present
    if (mh == true) && (mt == true)
        mask_bg = ~(mask_hand | mask_trace);
        m1 = times(img, mask_hand);
        m2 = times(img, mask_trace);
    elseif (mh == true) && (mt == false)
        mask_bg = ~(mask_hand);
        m1 = times(img, mask_hand);
    elseif (mh == false) && (mt == true)
        mask_bg = ~(mask_trace);
        m2 = times(img, mask_trace);
    else
        fprintf('No hand or trace mask found.\n');
    end
    
    m3 = times(img, mask_bg);
    % Reformat 3D array into a 2D
    m1 = reshape(m1,size(m1,1)*size(m1,2),size(m1,3));
    m2 = reshape(m2,size(m2,1)*size(m2,2),size(m2,3));
    m3 = reshape(m3,size(m3,1)*size(m3,2),size(m3,3));
    
    % Remove zero pixels
    m1(all(m1==0,2),:)=[];
    m2(all(m2==0,2),:)=[];
    m3(all(m3==0,2),:)=[];
end

function [segment] = bayes(TI, m1, m2, m3)
    % Classification matrix to be output
    segment = zeros(size(TI,1),size(TI,2));
    % Reshape the Test Image into an Nx3 Matrix
    testImgVect = reshape(TI,size(TI,1)*size(TI,2),size(TI,3));
    tLen = length(testImgVect);
    % Create labels to pair with the classes when combined into a single
    % Matrix
    len1 = length(m1);
    len2 = len1 + length(m2);
    len3 = len2 + length(m3);
    
    % Three classes, three labels - Hand, Trace, and Background
    labels = {};
    for i = 1:len1
        labels{i} = 'Hand';
    end
    
    for i  = len1:len2
        labels{i} = 'Trace';
    end
    
    for i = len2:len3
        labels{i} = 'Background';
    end
    
    ms = vertcat(m1,m2,m3);
    % Get the prioris of each class
    prior = my_priori(m1,m2,m3);
    % ms is a vector of all classes combined with the corresponding
    % vector label
    bayesClass = fitcnb(ms,labels,...
    'ClassNames',{'Hand', 'Trace', 'Background'}, 'Prior', prior);
    
    w = waitbar(0,'Progress Bar');
    % Parse through every test image pixel and predict its label
    for i = 1:tLen
        pixel = testImgVect(i,:);   
        label = predict(bayesClass,pixel);
        
        if (strcmpi(label, 'Hand'))
            class = 1;
        elseif (strcmpi(label, 'Trace'))
            class = 2;
        else
            class = 3;
        end

        segment(i) = class;
        % Update the wait bar after a pixel has been classified
        w = waitbar(i/tLen,w);
    end;
end

function [prior] = my_priori(x1, x2, x3)
    x1Len = length(x1);
    x2Len = length(x2);
    x3Len = length(x3);
    total = (x1Len + x2Len + x3Len);
    
    pri1 = x1Len / total;
    pri2 = x2Len / total;
    pri3 = x3Len / total;
    
    prior = [pri1, pri2, pri3];
end

function [h] = plot_2D_gauss(mu, sd)
    ix = -2*sd:1:5*sd;
    iy = pdf('normal', ix, mu, sd);
    h = iy;
    plot(ix,iy);
end

function [gaus] = gauss_distribution(x, mu, sigma)
    p1 = -.5 * ((x - mu)/sigma) .^ 2;
    p2 = (sigma * sqrt(2*pi));
    gaus = exp(p1) ./ p2; 
end