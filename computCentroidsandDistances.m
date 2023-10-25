function [centroids distances] = computCentroidsandDistances(X, idx, K)
%calculate centroids of each cluster, and sum of distances within each cluster
[m n] =size(X);
centroids = zeros(K,n);
distances = 0;

for i = 1:K
    xi = X(idx == i, :);
    ck = size(xi, 1);
    %centroids(i, :) = (1/ck)*[sum(xi(:,1)) sum(xi(:,2))];
    centroids(i,:) = mean(xi);
    %d = 0; % distance =0
    %for j = 1:ck
        %d = d + sqrt(sum((xi(j,:) - centroids(i,:)).^2)); %this is the
        %distance within a cluster
    %end
    %distances1 = d + distances1; %this is sum of distances among all clusters.
    t = (xi - mean(xi)).^2;
   distances = distances + sum(sqrt(t(:,1) + t(:,2))); %this is to calculate the sum of distances among all clusters
end
end

