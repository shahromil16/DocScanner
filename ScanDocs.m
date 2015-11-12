function [corners] = ScanDocs(im)

fns = im;
I = im2double(fns);
% imread does not use the EXIF rotation, so we manually rotate
I = permute(I,[2 1 3]); 
I = I(:,end:-1:1,:);
%a = size(rgb2gray(fns));
fsize = 1200;
[p] = rectifyDocument(I,fsize);
[corners] = abs([p]);
end

function [pq] = rectifyDocument(I,fixedSize) % input: RGB image as double

% Downscale image to fixed size in order to reduce the influence of 
% noise and small-scale texture
imHeight = size(I,1);
imWidth = size(I,2);
imDiag = sqrt(imWidth^2 + imHeight^2);
downScale = 1
I = imresize(I,1,'bicubic');
J = rgb2gray(I);
H = im2bw(J);

% Detect the 4 dominant lines in the image
BW = edge(H,'canny',[],5);
[H,T,R] = hough(BW);
P = houghpeaks(H,4,'Threshold',0,'NHoodSize',[ceil(fixedSize/4)+1, 41]);
theta = T(P(:,2));
rho = R(P(:,1));
if (length(theta) < 4)
    fprintf('Could not find all lines\n');
    return;
end

% Display detected lines
figure;
subplot(1,2,1);
imagesc(I(:,:,2));
shading flat; hold on; colormap gray; axis equal;
for i = 1:length(theta)
    t = theta(i) / 180 * pi;
    r = rho(i);
    if (abs(t) > pi/4)
        u = 0:size(J,2);
        v = 1 + (r - u*cos(t) )/ sin(t);    
    else
        v = 0:size(J,1);
        u = 1 + (r - sin(t)*v) / cos(t);
    end
    plot(u,v,'g-','LineWidth',2);
end
% Compute line equations
% The Hough transform returns line equations of the form
%   cos(theta)*x + sin(theta)*y = rho
% We first sort the lines according to angle. Since the document is 
% approximately rectangular, this groups the lines into parallel
% lines
[~,order] = sort(abs(theta)); 
coefficients = [cos(theta' / 180 * pi), sin(theta' / 180 * pi), -rho'];
L1 = coefficients(order(1),:); % start with smallest angle
L2 = coefficients(order(2),:); % parallel to L1
L3 = coefficients(order(3),:); % perpendicular to L1 and L2
L4 = coefficients(order(4),:); % parallel to L3

% Compute p1, p2, p3, p4 as intersections of L1,...,L4;
% see above figure.
p = zeros(4,3); 
p(1,:) = cross(L1,L3); % p1
p(2,:) = cross(L2,L3); % p2
p(3,:) = cross(L2,L4); % p3
p(4,:) = cross(L1,L4); % p4
p = p(:,1:2) ./ [p(:,3), p(:,3)];
% We now need to reorder the points such that they correspond to 
% the physical document vertices

% Sort points into clockwise order (see above figure)
v = p(2,:) - p(1,:); 
w = p(3,:) - p(1,:);
a = (v(1)*w(2) - w(1)*v(2)) / 2; % signed area of triangle (p1,p2,p3)
if (a < 0)
    p = p(end:-1:1,:); % reverse vertex order
end

% Make sure that first vertex p1 lies either topmost or leftmost,
% and lies at the start of a short side
edgeLen = [
    norm(p(1,:) - p(2,:));
    norm(p(2,:) - p(3,:));
    norm(p(3,:) - p(4,:));
    norm(p(4,:) - p(1,:))
    ];
sortedEdgeLen = sort(edgeLen);
startsShortEdge = (edgeLen' <= sortedEdgeLen(2)); % flag which ones are at a short edge
% find the one that is closest to top left corner of image
idx = find(startsShortEdge,2);
if (norm(p(idx(1),:)) < norm(p(idx(2),:)))
    p1Idx = idx(1);
else
    p1Idx = idx(2);
end
order = mod(p1Idx - 1 : p1Idx + 2, 4) + 1; % reorder to start at right vertex
p = p(order,:);

% Diplay vertex order
for i = 1:4
    text(p(i,1),p(i,2),sprintf('%d',i),'BackgroundColor','white');
end

% Define the document vertices in paper coordinates
% We use the A4 standard: 210x280mm
paperWidth = 210; 
paperHeight = 280;
q = [
    0,          0;               % corresponds to p1
    paperWidth, 0;               % corresponds to p2
    paperWidth, paperHeight;     % corresponds to p3
    0,          paperHeight;     % corresponds to p4
];

% Compute homography p -> q
H = computeHomography(p,q);

% Modify homography to trim the edges by 2% (slight enlargement)
trimPercentage = 2;
eps = 1 / (1 - trimPercentage/100) - 1;
trimEdges = [
    1+eps, 0,    -eps*paperWidth/2;
    0,     1+eps, -eps*paperHeight/2;
    0,     0,     1;
];
H = trimEdges * H;

% Warp original image to paper coordinates
h = size(I,1);
w = size(I,2);
u = repmat(1:w,[h 1]);
v = repmat(1:h,[w 1])';
uv = [u(:),v(:),ones(length(u(:)),1)];
uv = uv*H';
uv = uv(:,1:2) ./ [uv(:,3),uv(:,3)];
u = reshape(uv(:,1),h,w);
v = reshape(uv(:,2),h,w);
subplot(1,2,2);
surf(u,v,zeros(h,w),I); 
shading interp; 
colormap gray; 
axis equal;
axis([0 paperWidth 0 paperHeight]);
set(gca,'YDir','reverse');
pq = p;
end

function H = computeHomography(p,q)
% Quick implementation of homography computation
% There are better ways to do this, see Hartley & Zisserman's book
A = [p(:,1),p(:,2),ones(4,1),zeros(4,3),-p(:,1).*q(:,1),-p(:,2).*q(:,1);
     zeros(4,3),p(:,1),p(:,2),ones(4,1),-p(:,1).*q(:,2),-p(:,2).*q(:,2)];
b = [q(:,1);q(:,2)];    
Hvec = A\b;
H = reshape([Hvec;1],3,3)';
end