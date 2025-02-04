% Compute LLE Embedding
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
