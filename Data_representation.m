clear;
clc;
tic;

% Load the data.
data = load("balanced_animal_data.mat");

% Read the images and labels from the data.
images = data.data;
labels = data.labels;

% Initialize a figure for plotting
figure;

% Loop through each class
for class_idx = 1:10
    % Find indices of images belonging to the current class
    class_indices = find(labels == class_idx);
    
    % Select one image randomly from the current class
    selected_image_index = class_indices(randi(length(class_indices)));
    
    % Get the selected image
    selected_image = images(selected_image_index, :, :, :);
    
    % Reshape the image to 64x64x3
    reshaped_image = reshape(selected_image, [64, 64, 3]);
    
    % Plot the image
    subplot(2, 5, class_idx);
    imshow(reshaped_image);
    title(['Class ', num2str(class_idx)]);
end

% Reshape images into a 2D matrix
num_images = size(images, 1);
num_pixels = size(images, 2) * size(images, 3) * size(images, 4);
reshaped_images = reshape(images, num_images, num_pixels);

% converting data into double
all_images = double(reshaped_images);
labels = labels';

% Making them zero centered
% Calculate the mean of all images
mean_image = mean(all_images, 1);

% Subtract the mean from each image
zero_centered_images = all_images - mean_image;

% Calculate the standard deviation of each pixel
std_dev = std(zero_centered_images, 0, 1);

% Divide each pixel by its standard deviation
z_scores_images = zero_centered_images ./ std_dev;

% Plot on raw data
figure;
gscatter(z_scores_images(:,1), z_scores_images(:,2), labels, "filled");
title('Raw data');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Location', 'best');


% BUILDING KNN CLASSIFIER ON RAW DATA
% Define the proportions for train, validation, and test sets
train_ratio = 0.8;  % 80% for training
val_ratio = 0.10;   % 10% for validation
test_ratio = 0.10;  % 10% for testing

% Calculate the number of samples for each set
num_samples = size(all_images, 1);
num_train = floor(train_ratio * num_samples);
num_val = floor(val_ratio * num_samples);
num_test = num_samples - num_train - num_val;

% Shuffle the indices
indices = randperm(num_samples);

% Split the indices into train, validation, and test sets
train_indices = indices(1:num_train);
val_indices = indices(num_train+1:num_train+num_val);
test_indices = indices(num_train+num_val+1:end);

% Split the data based on the indices
train_images = all_images(train_indices, :, :, :);
train_labels = labels(train_indices, :);

val_images = all_images(val_indices, :, :, :);
val_labels = labels(val_indices, :);

test_images = all_images(test_indices, :, :, :);
test_labels = labels(test_indices, :);

% Define the number of neighbors (k) for the KNN classifier
k = 20;  

% Train the KNN classifier
knn_classifier = fitcknn(train_images, train_labels, 'NumNeighbors', k);

% Predict labels for the validation set
val_predicted_labels = predict(knn_classifier, val_images);

% Evaluate the accuracy on the validation set
[accuracy_val_raw, precision_val_raw, recall_val_raw, F1_score_val_raw] = calculate_metrics(val_labels, val_predicted_labels);
fprintf('Validation Accuracy (raw data): %.2f%%', accuracy_val_raw * 100);
fprintf('\nValidation F1 score (raw data): %.2f%', F1_score_val_raw);

% Predict labels for the test set
test_predicted_labels = predict(knn_classifier, test_images);

% Evaluate the accuracy and F1 score on the test set
[accuracy_test_raw, precision_test_raw, recall_test_raw, F1_score_test_raw] = calculate_metrics(test_labels, test_predicted_labels);
fprintf('\nTest Accuracy (raw data): %.2f%%', accuracy_test_raw * 100);
fprintf('\nTest F1 score (raw data): %.2f%\n', F1_score_test_raw);

raw_data_time_taken = toc;
fprintf('Time taken for knn on raw data: %f\n', raw_data_time_taken);


% USING PCA TO REDUCE DIMENSIONS AND THEN BUILDING A CLASSIFIER
tic;
% Calculating PCA
[pca_transformed, mapping_pca] = compute_mapping(z_scores_images, "PCA", 6000);
pca_time = toc;
fprintf('Time taken for pca: %f\n', pca_time);


% Calculate the explained variance
explained_variance = mapping_pca.lambda / sum(mapping_pca.lambda);

% Plotting the explained variance
figure;
plot(cumsum(explained_variance), 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('Explained Variance by Principal Components PCA');
grid on;

% Plot the 2D Embeddings
figure;
scatter(pca_transformed(:, 1), pca_transformed(:, 2), 10, labels, "filled");
title('2D plot PCA');
xlabel('Dimension 1');
ylabel('Dimension 2');
colorbar;

% Initialize variables to store cumulative explained variance
cumulative_explained_variance = zeros(1, numel(400:50:1500));
accuracy_values = zeros(1, numel(400:50:1500));
idx = 1;

for n_components = 400:50:1500
    % Select the first n_components principal components
    num_components = n_components;
    transformed_data_pca = pca_transformed(:, 1:num_components);
    fprintf('\nNumber of components: %f\n', num_components);

    % Calculate explained variance
    explained_variance = sum(mapping_pca.lambda(1:num_components)) / sum(mapping_pca.lambda);
    cumulative_explained_variance(idx) = explained_variance;

    % Split the data
    train_images_pca = transformed_data_pca(train_indices, :);
    train_labels_pca = labels(train_indices);
    val_images_pca = transformed_data_pca(val_indices, :);
    val_labels_pca = labels(val_indices);
    test_images_pca = transformed_data_pca(test_indices, :);
    test_labels_pca = labels(test_indices);

    % Define the number of neighbors (k) for the KNN classifier
    k = 20;  

    % Train the KNN classifier
    knn_classifier_pca = fitcknn(train_images_pca, train_labels_pca, 'NumNeighbors', k);

    % Predict labels for the validation set
    val_predicted_labels_pca = predict(knn_classifier_pca, val_images_pca);

    % Evaluate the accuracy on the validation set
    [accuracy_val_pca, precision_val_pca, recall_val_pca, F1_score_val_pca] = calculate_metrics(val_labels_pca, val_predicted_labels_pca);
    fprintf('Validation Accuracy (PCA data): %.2f%%', accuracy_val_pca * 100);
    fprintf('\nValidation F1 score (PCA data): %.2f%', F1_score_val_pca);

    % Predict labels for the test set
    test_predicted_labels_pca = predict(knn_classifier_pca, test_images_pca);

    % Evaluate the accuracy and F1 score on the test set
    [accuracy_test_pca, precision_test_pca, recall_test_pca, F1_score_test_pca] = calculate_metrics(test_labels_pca, test_predicted_labels_pca);
    fprintf('\nTest Accuracy (PCA data): %.2f%%', accuracy_test_pca * 100);
    fprintf('\nTest F1 score (PCA data): %.2f%\n', F1_score_test_pca);
    % Store accuracy values
    accuracy_values(idx) = accuracy_test_pca;

    pca_data_time_taken = toc;
    fprintf('\nTime taken for knn on PCA data: %f\n', pca_data_time_taken);

    % Increment index
    idx = idx + 1;
end


% Plot accuracy against the number of components
figure;
plot(400:50:1500, accuracy_values, '-o', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Accuracy (%)');
title('Accuracy vs. Number of Components PCA');
grid on;

% Plot cumulative explained variance against the number of components
figure;
plot(400:50:1500, cumulative_explained_variance, '-s', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('Explained Variance vs. Number of Components PCA');
grid on;



% USING KPCA TO REDUCE DIMENSIONS AND THEN BUILDING A CLASSIFIER

tic;
kpca_transformed = compute_mapping(z_scores_images, "KernelPCA");
kpca_time = toc;
fprintf('Time taken for kpca: %f\n', kpca_time);

% Calculate the covariance matrix of the transformed data
covariance_matrix = cov(kpca_transformed);

% Compute the eigenvalues of the covariance matrix
eigenvalues = eig(covariance_matrix);

% Calculate explained variance for each component
total_variance = sum(eigenvalues);
explained_variance = eigenvalues / total_variance;

% Plot the explained variance
plot(1:length(explained_variance), explained_variance, 'bo-');
xlabel('Component Number');
ylabel('Explained Variance');
title('Scree Plot of KPCA');
grid on;
xlim([1, min(50, length(explained_variance))]); % Adjust the range as needed

% Converting data into real numbers
kpca_transformed_real = abs(kpca_transformed);

% Plot the 2D Embeddings
figure;
scatter(kpca_transformed(:, 1), kpca_transformed(:, 2), 10, labels, "filled");
title('2D plot KPCA');
xlabel('Dimension 1');
ylabel('Dimension 2');
colorbar;

% Plot the 2D Embeddings
figure;
scatter(kpca_transformed(:, 1), kpca_transformed(:, 2), 10, labels);
title('KPCA');
colorbar;

% Split the data based on the indices
train_images_kpca = kpca_transformed_real(train_indices, :, :, :);
train_labels_kpca = labels(train_indices, :);

val_images_kpca = kpca_transformed_real(val_indices, :, :, :);
val_labels_kpca = labels(val_indices, :);

test_images_kpca = kpca_transformed_real(test_indices, :, :, :);
test_labels_kpca = labels(test_indices, :);

% Define the number of neighbors (k) for the KNN classifier
k = 20;  

% Train the KNN classifier
knn_classifier_kpca = fitcknn(train_images_kpca, train_labels_kpca, 'NumNeighbors', k);

% Predict labels for the validation set
val_predicted_labels_kpca = predict(knn_classifier_kpca, val_images_kpca);

% Evaluate the accuracy on the validation set
[accuracy_val_kpca, ~, ~, F1_score_val_kpca] = calculate_metrics(val_labels_kpca, val_predicted_labels_kpca);
fprintf('Validation Accuracy (KPCA data): %.2f%%', accuracy_val_kpca * 100);
fprintf('\nValidation F1 score (KPCA data): %.2f%', F1_score_val_kpca);

% Predict labels for the test set
test_predicted_labels_kpca = predict(knn_classifier_kpca, test_images_kpca);

% Evaluate the accuracy and F1 score on the test set
[accuracy_test_kpca, ~, ~, F1_score_test_kpca] = calculate_metrics(test_labels_kpca, test_predicted_labels_kpca);
fprintf('\nTest Accuracy (KPCA data): %.2f%%', accuracy_test_kpca * 100);
fprintf('\nTest F1 score (KPCA data): %.2f%\n', F1_score_test_kpca);



% USING LLE TO REDUCE DIMENSIONS AND THEN BUILDING A CLASSIFIER
% Compute LLE 
tic;
lle_transformed = compute_mapping(all_images, 'LLE', 10);
lle_time = toc;
fprintf('Time takenfor LLE: %.2f seconds\n', lle_time);

% Plot 2D embedding
figure;
scatter(lle_transformed(:,1), lle_transformed(:,2), 10, 'filled');
title('2D Embedding LLE');
xlabel('Dimension 1');
ylabel('Dimension 2');

% % Plot the 2D Embeddings
% figure;
% scatter(lle_transformed(:, 1), lle_transformed(:, 2), 10, labels);
% title('2D plot LLE');
% xlabel('Dimension 1');
% ylabel('Dimension 2');
% colorbar;

% Converting data into real numbers
lle_transformed_real = abs(lle_transformed);

% Define the sizes of the train, validation, and test sets
train_ratio = 0.8;
val_ratio = 0.1;
test_ratio = 0.1;

% Get the number of samples
num_samples = size(z_scores_images, 1);

% Generate indices for the data based on the size of lle_transformed_real
indices = randperm(size(lle_transformed_real, 1));

% Split the indices based on the defined ratios
train_indices = indices(1:round(train_ratio * num_samples));
val_indices = indices(round(train_ratio * num_samples) + 1 : round((train_ratio + val_ratio) * num_samples));
test_indices = indices(round((train_ratio + val_ratio) * num_samples) + 1 : end);


% Split the data and labels based on the generated indices
train_images_lle = lle_transformed_real(train_indices, :);
train_labels_lle = labels(train_indices, :);

val_images_lle = lle_transformed_real(val_indices, :);
val_labels_lle = labels(val_indices, :);

test_images_lle = lle_transformed_real(test_indices, :);
test_labels_lle = labels(test_indices, :);


% Define the number of neighbors (k) for the KNN classifier
k = 20;  

% Train the KNN classifier
knn_classifier_lle = fitcknn(train_images_lle, train_labels_lle, 'NumNeighbors', k);

% Predict labels for the validation set
val_predicted_labels_lle = predict(knn_classifier_lle, val_images_lle);

% Evaluate the accuracy on the validation set
[accuracy_val_lle, ~, ~, F1_score_val_lle] = calculate_metrics(val_labels_lle, val_predicted_labels_lle);
fprintf('Validation Accuracy (LLE data): %.2f%%\n', accuracy_val_lle * 100);
fprintf('Validation F1 score (LLE data): %.2f%\n', F1_score_val_lle);

% Predict labels for the test set
test_predicted_labels_lle = predict(knn_classifier_lle, test_images_lle);

% Evaluate the accuracy and F1 score on the test set
[accuracy_test_lle, ~, ~, F1_score_test_lle] = calculate_metrics(test_labels_lle, test_predicted_labels_lle);
fprintf('\nTest Accuracy (LLE data): %.2f%%', accuracy_test_lle * 100);
fprintf('\nTest F1 score (LLE data): %.2f%\n', F1_score_test_lle);


% FUNCTION TO CALCULATE THE METRICS
function [accuracy, precision, recall, F1_score] = calculate_metrics(true_labels, predicted_labels)
    % Calculate accuracy
    accuracy = sum(predicted_labels == true_labels) / numel(true_labels);

    % Calculate confusion matrix
    C = confusionmat(true_labels, predicted_labels);

    % Calculate precision and recall for each class
    precision = zeros(1, size(C, 1));
    recall = zeros(1, size(C, 1));
    for i = 1:size(C, 1)
        true_positive = C(i, i);
        false_positive = sum(C(:, i)) - true_positive;
        false_negative = sum(C(i, :)) - true_positive;

        % Check for division by zero
        if true_positive == 0 && (false_positive > 0 || false_negative > 0)
            precision(i) = 0;
            recall(i) = 0;
        else
            precision(i) = true_positive / (true_positive + false_positive);
            recall(i) = true_positive / (true_positive + false_negative);
        end

    end

    % Calculate F1 Score
    F1_score = 2 * (precision .* recall) ./ (precision + recall);

    % Replace NaN values with 0
    F1_score(isnan(F1_score)) = 0;
end
