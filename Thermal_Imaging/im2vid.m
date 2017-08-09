workingDir = '\videos';
outputVideo = VideoWriter(fullfile(workingDir,'out.avi'));
outputVideo.FrameRate = 10;
open(outputVideo);
totalFrames = 42;

for i = 1:totalFrames
   j = ceil(i/2);
   imgName = sprintf('seqA%i.png',j);
   img = imread(imgName);
   writeVideo(outputVideo,img)
end

close(outputVideo);