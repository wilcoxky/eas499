%% Read Video
videoReader = vision.VideoFileReader('./squat_example.mov');  

%% Create Video Player
videoPlayer = vision.VideoPlayer;
% fgPlayer = vision.VideoPlayer;

%% Create Background Detector  (Background Subtraction)
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 50);

%% Create ShapeInserter
shapeInserter = vision.ShapeInserter('BorderColor','White');

%% Create Point tracker
pointTracker = vision.PointTracker;

%% Blob Analysis
% blob = vision.BlobAnalysis(...
%        'CentroidOutputPort', false, 'AreaOutputPort', false, ...
%        'BoundingBoxOutputPort', true, ...
%        'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 100);
bodyDetector = vision.CascadeObjectDetector('UpperBody');
   bodyDetector.MinSize = [60 60];
   bodyDetector.MergeThreshold = 10;

%% learn background and skip beginning
for i = 1:270
    I = step(videoReader);
    videoFrame = imrotate(I, -90);
    foreground = step(foregroundDetector,videoFrame);
    if i == 270
        figure, imshow(videoFrame);
        h0 = impoint(gca,[]);
        p0 = wait(h0)
        h1 = impoint(gca,[]);
        p1 = wait(h1)
        h2 = impoint(gca,[]);
        p2 = wait(h2)
        h3 = impoint(gca,[]);
        p3 = wait(h3)
        h4 = impoint(gca,[]);
        p4 = wait(h4)
        features = [p0; p1; p2; p3; p4];
        initialize(pointTracker, features, videoFrame)
    end
end

%% Loop through video
while  ~isDone(videoReader)
    %Get the next frame
    I = step(videoReader);
    videoFrame = imrotate(I, -90);
    [points,point_validity,scores] = step(pointTracker,videoFrame);
%     impoint
%     Detect foreground pixels
%     foregroundMask = step(foregroundDetector,videoFrame);
    % Perform morphological filtering
%     cleanForeground = imopen(foregroundMask, strel('Disk',1));
            
    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
%       bbox = step(blob, foregroundMask);
%      bboxBody = step(bodyDetector, videoFrame);

     % Draw bounding boxes around the detected cars
%       result = insertShape(videoFrame, 'Rectangle', bboxBody, 'Color', 'green');


      % Display tracked points
     videoFrame = insertMarker(videoFrame, points, '+', ...
            'Color', 'white');
     line0 = [points(1,1:2), points(2,1:2)];
     line1 = [points(2,1:2), points(3,1:2)];
     line2 = [points(3,1:2), points(4,1:2)];     
     line3 = [points(4,1:2), points(5,1:2)];
     u = points(4,1:2);
     v = points(5,1:2);
     CosTheta = dot(u,v)/(norm(u)*norm(v));
     ThetaInDegrees = acosd(CosTheta)
     videoFrame = insertShape(videoFrame, 'Line', line0, 'LineWidth', 4, 'Color', 'Red');
     videoFrame = insertShape(videoFrame, 'Line', line1, 'LineWidth', 4, 'Color', 'Yellow');
     videoFrame = insertShape(videoFrame, 'Line', line2, 'LineWidth', 4, 'Color', 'Blue');     
     videoFrame = insertShape(videoFrame, 'Line', line3, 'LineWidth', 4, 'Color', 'Green');
    % Display output 
     step(videoPlayer, videoFrame);
%     step(fgPlayer,cleanForeground);

end

%% release video reader and writer
release(videoPlayer);
release(videoReader);
release(blob);
release(shapeInserter);
% release(fgPlayer);
delete(videoPlayer); % delete will cause the viewer to close
% delete(fgPlayer);