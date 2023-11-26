function [centroids, distances] = computeCentroidsandDistances(X, idx, K)
%calculate centroids of each cluster, and sum of distances within each cluster
[m, n] =size(X);
centroids = zeros(K,n);
distances = 0;

for i = 1:K
    xi = X(idx == i, :);
    centroids(i,:) = mean(xi);
    t = (xi - mean(xi)).^2;
   distances = distances + sum(sqrt(t(:,1) + t(:,2))); %this is to calculate the sum of distances among all clusters
end
end

