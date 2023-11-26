clc;
close all;
clear;
% Generating dataset
labels1 = ones(1,1000);
labels2 = repmat(2,1,1000);
labels3 = repmat(3,1,1000);
labels4 = repmat(4,1,1000);
initialLabels = [labels1, labels2, labels3, labels4];
Mean1 = [-3; -1];
Std1 = std(Mean1);
Mean2 = [3; 4];
Std2 = std(Mean2);
Mean3 = [4; -4];
Std3 = 1;
Mean4 = [-3; 4];
Std4 = 1;
samples = 1000; 
data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
data3 = Std3 * randn(2, samples) + repmat(Mean3, 1, samples);
data4 = Std4 * randn(2, samples) + repmat(Mean4, 1, samples);

% Plot original dataset
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

trainData = [data1 data2 data3 data4]; % creating training data

% K-Means from first principles
X = trainData';
K = 4;
max_iterations = 10; %5;
max_k = 10;
elbow_found = false;
k_sumD = zeros(max_k,1);
k_indices = zeros(4000,max_k);
k_mse = zeros(max_k,1);
final_centroids = zeros(max_k,2);
final_indices = zeros(max_k,1);
elbow = 0;
for k = 1:max_k % iterate through multiple K values 
    centroids = initCentroids(X,k);
    sumD = zeros(k,1);
    for i = 1:max_iterations
        indices = reassignPoints(X, centroids); % reassign the indices to the relevant clusters
        [centroids, distances] = computeCentroidsandDistances(X, indices, k); % recalculate centroids, and distances among all clusters
        sumD(i) = distances; 
    end
    k_mse(k) = mse(X, centroids)/k;
    k_sumD(k) = sumD(max_iterations); % Get final distance value for finished iteration of K
    k_indices(:,k) = indices; % Get final indices for finished iteration of K
    if k == 4
        final_centroids = centroids;
        final_indices = indices;
    end
end

% Plotting Calinski-Harabasz evaluation for K means
eva = evalclusters(X, k_indices, 'CalinskiHarabasz');
figure;
plot(eva);
title('Calinski-Harabasz Method for Optimal K');
grid on;

% Plotting Davies-Bouldin
evaDB = evalclusters(X, k_indices, 'DaviesBouldin');
figure;
plot(evaDB);
title('Davies-Bouldin Method for Optimal K');
grid on;

% Plotting each k_mse
figure;
plot(k_mse, 'bo-', 'LineWidth',2);
title('Elbow Method for Optimal K');
    xlabel('K value')
    ylabel('MSE');
    grid on;

% Plotting K-Means
message = sprintf('KMeans Clustering (MaxIterations = %d)', max_iterations);
figure;

plot(trainData(1, final_indices == 1),trainData(2, final_indices == 1),'b.', 'MarkerSize',12);% for 1st cluster
hold on;
plot(trainData(1, final_indices == 2), trainData(2, final_indices == 2), 'r.', 'MarkerSize',12); %for 2nd cluster
plot(trainData(1, final_indices == 3), trainData(2, final_indices == 3), 'y.', 'MarkerSize',12); %for 3rd cluster
plot(trainData(1, final_indices == 4), trainData(2, final_indices == 4), 'g.', 'MarkerSize',12); %for 4th cluster

plot(final_centroids(1,1),final_centroids(1,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 1st centroid
plot(final_centroids(2,1),final_centroids(2,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 2nd centroid
plot(final_centroids(3,1),final_centroids(3,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 3rd centroid
plot(final_centroids(4,1),final_centroids(4,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 4th centroid

title('K-Means clustering when K = 4')
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

hold off;


% Plotting GMM
figure;
hold on;
gm = fitgmdist(X,4);
scatter(X(:,1),X(:,2),10,'.');

gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);

idx = cluster(gm,X);

gscatter(X(:,1),X(:,2),idx);
fcontour(gmPDF,[[-10,10] [-8 12]]);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Location','best');
hold off;

% GMM Calinski-Harabasz
evaGMM = evalclusters(X, idx, 'CalinskiHarabasz');

% GMM Davies-Bouldin
evaGMMDB = evalclusters(X, idx, 'DaviesBouldin');


% Plot comparison of Calinski-Harabasz index values
figure;
hold on;
bar(1,[evaGMM.CriterionValues; eva.CriterionValues(1,4)]);
title('Calinski-Harabasz index values for GMM and K-means value (K = 4)');
xlabel('Model');
ylabel('Calinski-Harabasz index')
ylim([12000, 14500])
legend('GMM', 'K-means')
hold off;

% Plot comparison of Davies-Bouldin index values
figure;
hold on;
bar(1,[evaGMMDB.CriterionValues; evaDB.CriterionValues(1,4)]);
title('Davies-Bouldin index values for GMM and K-means value (K = 4)');
xlabel('Model');
ylabel('Davies-Bouldin index')
ylim([0.47, 0.5])
legend('GMM', 'K-means')
hold off;