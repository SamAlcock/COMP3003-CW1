function centroids = initCentroids(X,K)
centroids = zeros(K, size(X,2)); % initialise centroid matrix
randidx = randperm(size(X,1)); 
centroids = X(randidx(1:K), :); 
end

