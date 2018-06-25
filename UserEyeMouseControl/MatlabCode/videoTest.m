inputV = VideoReader('outpytenfps.avi');
outputV = VideoWriter('wow.avi'); %create the video object
outputV.FrameRate = inputV.FrameRate;
numFrames = inputV.NumberOfFrames;
open(outputV); %open the file for writing

se = strel('disk',4);                   %Create disk with radius 4 pixels and is it in openIm to remove those less than that radius

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%--------COULD USE IMPROVEMENT ESPECIALLY WHEN CROPPING EYE REGION---------
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for n = 2:numFrames-1
    %Using the difference between frames we will extract the eye region
    %when the user involuntarily blinks which will make a noticeable
    %difference between the difference between the two images
    f1 = im2double(rgb2gray(read(inputV,n)));
    f2 = im2double(rgb2gray(read(inputV,n+1)));
    ff = (f2-f1);                     %Difference between frames
    openIm = im2bw(imopen(imgaussfilt(ff,7),se),0.04);              %Gets rid of noise
    
    stats = regionprops('table',openIm,'Centroid','BoundingBox');

    centr = stats.Centroid;
    %make sure we actually obtained only two objects --> two eyes
    if(size(centr) == [2,2]) 
        disp('Potential Eye Found');
        box = stats.BoundingBox;
        %Make sure we didn't extract a noisy eye or unrelated object
        eyeWidth = box(5);
        if(box(6) < (eyeWidth - 8) || box(6) > (eyeWidth+8))
            continue
        end
        if(eyeWidth < 15 && eyeWidth > 8)
            newBound1 = [centr(1,1:2)-4*(box(1,3:4)),box(1,3:4)*8];
            newBound2 = [centr(2,1:2)-4*(box(1,3:4)),box(1,3:4)*8];
        elseif (eyeWidth >= 15 && eyeWidth < 25)
            newBound1 = [centr(1,1:2)-3*(box(1,3:4)),6*box(1,3:4)];
            newBound2 = [centr(2,1:2)-3*(box(1,3:4)),6*box(1,3:4)];   
        elseif (eyeWidth >= 25 && eyeWidth < 40)
            newBound1 = [centr(1,1:2)-2*(box(1,3:4)),4*box(1,3:4)];
            newBound2 = [centr(2,1:2)-2*(box(1,3:4)),4*box(1,3:4)]; 
        elseif (eyeWidth >= 40)
            newBound1 = [centr(1,1:2)-(box(1,3:4)),3*box(1,3:4)];
            newBound2 = [centr(2,1:2)-(box(1,3:4)),3*box(1,3:4)];
        else
            continue;
        end
        disp('Eye Found!');
        break;
    end
end

%Calibration

%Look at the middle of the screen:
%   This calibration point will set a center line for the pupils.  We will 
%   be extracting both eyes since depending on whether the subject
%   is looking left or right relative to the camera, one eye has better
%   pupil extraction then the other.  The inner part of the eye nearest to
%   the nose produces unecessary noise do to shadows.  For each measurment 
%   [px,py] we capture the x-coordinate and y-coordinate of the center of
%   the pupil found respectively.

midRE = [40,31];
midLE = [28,27];

%Look at the top left corner, bottom left corner, top right corner, and
%bottom right corner:
%   Now we will want to extract the corners of the screen for the left and
%   right pan respectively

leftUpLE = [32,22];
leftBotLE = [32,38];

rightTopRE = [50,22];
rightBotRE = [50,39];

%Using the center line already found we will complete the box (four corner
%coordinates for each eye given the middle line)

midUpLE = [midLE(1),leftUpLE(2)];
midBotLE = [midLE(1),leftBotLE(2)];

midUpRE = [midRE(1),leftUpLE(2)];
midBotRE = [midRE(1),leftBotLE(2)];

%Normalize the left eye box and right eye box so the top left corner is set
%to [0,0] (alternatively [xMin-xMin,yMin-yMin]) and the bottom right corner
%is set to [xMax-xMin,yMax-yMin]

lBoxMax = midBotLE - leftUpLE; 
rBoxMax = rightBotRE - midUpRE;
midNormLE = midLE - leftUpLE;
midNormRE = midRE - midUpRE;


%Now let's find the coefficient between the screen resolution and our eye
%boxes.  Assume standard 1920 x 1200 monitor. Use 960 since we have left
%monitor region and right monitor region for left and right eye
%respectively

monRes = [960,1200]; 
resFactorL = monRes ./ lBoxMax;
resFactorR = monRes ./ rBoxMax;


%Eye region is now found but depending on the width of the original eye
%image we will apply a different filter to extract the pupil
%Note: the left (L) and right (R) are from the perspective of the person
%i.e. the anatomical left and right eye respectively (which also
%corresponds to the left and right of the computer screen as a normal user
%views it

screenPix = [];

for n = 2:numFrames-1
    r = rgb2gray(imcrop(read(inputV,n),newBound1));
    l = rgb2gray(imcrop(read(inputV,n),newBound2));
    
    normR = im2uint8(mat2gray(r));
    normL = im2uint8(mat2gray(l));

    binR = im2bw(imgaussfilt(normR,1),0.1);
    binL = im2bw(imgaussfilt(normL,1),0.1);    
    
    %Normalize to mitigate brightness issues
    resR = normR.*(im2uint8(binR));
    resL = normL.*(im2uint8(binL));
    
    %Using an average convolution to extract the minimum
    %point of intensity (assuming the center of the pupil is the darkest region)
    avgKern = ones(10,10)/100;

    avgR = imfilter(resR,avgKern);
    avgL = imfilter(resL,avgKern);

    avgR(:,size(avgR,2)-5 : size(avgR,2)) = [];
    avgR(size(avgR,1)-5 : size(avgR,1),:) = [];
    avgR(1:5,:) = [];
    avgR(:,1:5) = [];
    avgL(:,size(avgL,2)-5 : size(avgL,2)) = [];
    avgL(size(avgL,1)-5 : size(avgL,1),:) = []; 
    avgL(1:5,:) = [];
    avgL(:,1:5) = [];   

    [~,indLowL] = min(avgL(:));
    [~,indLowR] = min(avgR(:));

    [lx,ly] = ind2sub(size(avgL),indLowL);
    [rx,ry] = ind2sub(size(avgR),indLowR);

    writeImRA = insertMarker(normR,[ry+5,rx+5]);  %The amount of time it took me to realize I needed to add the offset from chopping off part of avgR was ridiculous
    writeImLA = insertMarker(normL,[ly+5,lx+5]);
    %imshowpair(avgL,writeImLA,'montage')
 
    %Using region processing to extract pupil region and calulate centroid of it 
    
    statsPupR = regionprops('table',not(binR),'Centroid','Area');
    statsPupL = regionprops('table',not(binL),'Centroid','Area');
    [~,ind1] = max(statsPupR.Area);
    eyeCentR = ceil(statsPupR.Centroid(ind1,:));
    [~,ind2] = max(statsPupL.Area);
    eyeCentL = ceil(statsPupL.Centroid(ind2,:));
    if(isempty(eyeCentR))
        eyeCentR = [1,1];
        disp('Lost the location of the Right eye');
    end
    if(isempty(eyeCentL))
        eyeCentL = [1,1];
        disp('Lost the location of the Left eye');
    end    
    writeImRC = insertMarker(normR,eyeCentR);
    writeImLC = insertMarker(normL,eyeCentL);
    
    %Make a hybrid center-pupil locator using the centroid and least
    %luminous point
    hybridR = (eyeCentR+[ry+5,rx+5])./2; 
    hybridL = (eyeCentL+[ly+5,lx+5])./2; 
    hybImR = insertMarker(r,hybridR);
    hybImL = insertMarker(l,hybridL);
    
    writeIm = cat(2,hybImR,hybImL);

    writeVideo(outputV,writeIm);
%     
%     if((eyeCentR - midNormRE)> 0)
%         %Use right eye
%         screenPos = [960,0] + ((eyeCentR-midUpRE).*resFactorR);
%         screenPix = [screenPix;screenPos];
%     else
%         %Use left eye
%         screenPos = ((eyeCentL-leftUpLE).*resFactorL);
%         screenPix = [screenPix;screenPos];
%     end
    
    
end

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%!!!!!!!!!!!!!!!!TODO: Voluntary blink detection!!!!!!!!!!!!!!!!!!!!!!!!!!!
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%https://arxiv.org/pdf/1202.6517.pdf -- eye tracking

close(outputV);
