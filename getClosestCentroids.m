function indices = getClosestCentroids(X, centroids)
%for each datapoint x, calculate which cluster it belongs to, and put
%result in the indices matrix (Nx1), e.g. for 1st datapoint, indices(1) can be 1 or 2 (for 2 clusters).  
K = size(centroids, 1);
indices = zeros(size(X,1), 1);
m = size(X,1);

for i = 1:m %for loop to go through all data points
    k = 1;
   % min_dist = sum((X(i,:) - centroids(1,:)) .^ 2); 
   % there are two ways to calculate min_dist
   min_dist = (X(i,:) - centroids(1,:)) * (X(i,:) - centroids(1,:))';
    for j = 2:K
        dist = sum((X(i,:) - centroids(j,:)) .^ 2);
        if (dist < min_dist)
            min_dist = dist;
            k = j; % k is the cluster the datapoint belongs to
        end
    end
    indices(i) = k;
end
end

