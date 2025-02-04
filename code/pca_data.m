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

% Function to calculate metrics
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
            precision(i) = true_positive / (true_positive + false_positive);
            recall(i) = true_positive / (true_positive + false_negative);
        end

        % Calculate F1 Score
        F1_score = 2 * (precision .* recall) ./ (precision + recall);
    end