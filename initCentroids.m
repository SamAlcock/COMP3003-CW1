function centroids = initCentroids(X,K)
%INITCENTROIDS 
centroids = zeros(K, size(X,2));
randidx = randperm(size(X,1)); %random permutation of the integers of 1:N (here N is the sample size of X).
centroids = X(randidx(1:K), :); % centroids are datapoints corresponding to those random K positions.
end

