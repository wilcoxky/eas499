% videoReader = vision.VideoFileReader('./squat_example.mov');  
videoSource = vision.VideoFileReader('./squat_example.mov','ImageColorSpace','Intensity','VideoOutputDataType','uint8');
%%Background detector module.
detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 90, ... % 5 because of short video
       'InitialVariance', 30*30); % initial standard deviation of 30
blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 1000);
shapeInserter = vision.ShapeInserter('BorderColor','White');
bodyDetector = vision.CascadeObjectDetector('UpperBody');
   bodyDetector.MinSize = [60 60];
   bodyDetector.MergeThreshold = 10;
videoPlayer = vision.VideoPlayer();
se = strel('square', 600); % morphological filter for noise removal
while ~isDone(videoSource)
     frame  = step(videoSource);
     bboxBody = step(bodyDetector, frame);
     IBody = insertObjectAnnotation(frame, 'rectangle',bboxBody,'Face ans Shoulder');
     %% Detect Background. Out is mask 
     fMask = step(detector, frame);
     fgMask = imopen(fMask, se);
      h = fspecial('average', 5);
     rgb2 = imfilter(fMask, h);
   
     bbox   = step(blob, rgb2);
     out    = step(shapeInserter, rgb2, bbox); % draw bounding boxes around cars
     step(videoPlayer, out); % view results in the video player
end
release(videoPlayer);
release(videoSource);