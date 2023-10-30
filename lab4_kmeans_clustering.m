% Generate a 2D uncorrelated dataset
% Implement KMeans from first principles 
%
clc;
close all;
clear;
% 1. Generate dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mean1 = [-4; -1];
Std1 = std(Mean1);
Mean2 = [3; 4];
Std2 = std(Mean2);
Mean3 = [4; -4];
Std3 = 1;
Mean4 = [-3; 5];
Std4 = 1;
samples = 1000; 
data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
data3 = Std3 * randn(2, samples) + repmat(Mean3, 1, samples);
data4 = Std4 * randn(2, samples) + repmat(Mean4, 1, samples);
figure;
plot(data1(1, :), data1(2,:),'b.', 'MarkerSize',12);
hold on;
plot(data2(1,:), data2(2,:), 'r.', 'MarkerSize',12);
plot(data3(1,:), data3(2,:), 'y.', 'MarkerSize',12);
plot(data4(1,:), data4(2,:), 'g.', 'MarkerSize',12);
legend('Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4', 'location', 'NW'); 
title('Original Dataset');
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

% 2.Concatenate the training datasets and plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData = [data1 data2 data3 data4];
figure;
plot(trainData(1, :), trainData(2,:),'b.', 'MarkerSize',12);
legend('Dataset','location', 'NW'); %'best');
title('Merged Dataset');
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

% 3. Implement KMeans from first principles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = trainData';
K = 4;
max_iterations = 10; %5;
max_k = 10;
% randomly assign the K data points as the initial "centroids" of the K clusters
elbow_found = false;
k_sumD = zeros(max_k,1);
k_mse = zeros(max_k,1);
final_centroids = zeros(max_k,2);
final_indices = zeros(max_k,1);
all_indices = zeros(max_k, 4000);
elbow = 0;
for k = 1:max_k % iterate through multiple K values 
    centroids = initCentroids(X,k);
    sumD = zeros(k,1);
    for i = 1:max_iterations
        indices = getClosestCentroids(X, centroids); % reassign the indices to the relevant clusters
        [centroids, distances] = computCentroidsandDistances(X, indices, k); % recalculate centroids, and distances among all clusters
        sumD(i) = distances; 
        k_mse(i) = mean_squared_error(distances);
        all_indices(k, :) = indices;

    end
    
    k_sumD(k) = k_mse(max_iterations); % Get final MSE for finished K value
    if k > 1 && k_sumD(k - 1) - 1000000 < k_sumD(k) && elbow_found == false
        elbow = k - 1;
        fprintf('Elbow found at k = %d\n', elbow)
        elbow_found = true;
        final_centroids = centroids();
        final_indices = all_indices(elbow, :);


    end
end
%idx = kmeans (X, K); %this is to use kmeans function to calculate the
%indices for each cluster
   figure;
   plot(sumD, 'bo-', 'linewidth',2);
   title('Overall distances vs. number of iterations');
    xlabel('Iterations');
    ylabel('Overall distances');
    grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting each k_sumD
figure;
plot(k_sumD, 'bo-', 'LineWidth',2);
title('Elbow Method for Optimal K');
    xlabel('K value')
    ylabel('MSE');
    grid on;

% 4. Run your KMeans function on the data and plot results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
message = sprintf('KMeans Clustering (MaxIterations = %d)', max_iterations);
figure;
fprintf("Elbow = %d\n", elbow);
for i = 1:elbow
    plot(trainData(1, final_indices == i), trainData(2, final_indices == i),'.', 'Color', [rand,rand,rand], 'MarkerSize',12);
    hold on;
    plot(final_centroids(i,1),final_centroids(i,2),'kx', 'MarkerSize',15,'LineWidth',3); 
end
hold off;

xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;