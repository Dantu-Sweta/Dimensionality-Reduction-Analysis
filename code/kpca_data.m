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
