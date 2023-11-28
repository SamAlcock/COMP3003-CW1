function indices = reassignPoints(X, centroids)
K = size(centroids, 1);
indices = zeros(size(X,1), 1);
m = size(X,1);

for i = 1:m 
    k = 1;
   min_dist = (X(i,:) - centroids(1,:)) * (X(i,:) - centroids(1,:))'; % calculate original minimum distance
    for j = 2:K
        dist = sum((X(i,:) - centroids(j,:)) .^ 2);
        if (dist < min_dist)
            min_dist = dist;
            k = j; 
        end
    end
    indices(i) = k;
end
end

