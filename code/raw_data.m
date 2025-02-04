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

% % Define the number of neighbors (k) for the KNN classifier
% k = 20;  
% 
% % Train the KNN classifier
% knn_classifier = fitcknn(train_images, train_labels, 'NumNeighbors', k);
% 
% % Predict labels for the validation set
% val_predicted_labels = predict(knn_classifier, val_images);
% 
% % Evaluate the accuracy on the validation set
% [accuracy_val_raw, precision_val_raw, recall_val_raw, F1_score_val_raw] = calculate_metrics(val_labels, val_predicted_labels);
% fprintf('Validation Accuracy (raw data): %.2f%%', accuracy_val_raw * 100);
% fprintf('\nValidation F1 score (raw data): %.2f%', F1_score_val_raw);
% 
% % Predict labels for the test set
% test_predicted_labels = predict(knn_classifier, test_images);
% 
% % Evaluate the accuracy and F1 score on the test set
% [accuracy_test_raw, precision_test_raw, recall_test_raw, F1_score_test_raw] = calculate_metrics(test_labels, test_predicted_labels);
% fprintf('\nTest Accuracy (raw data): %.2f%%', accuracy_test_raw * 100);
% fprintf('\nTest F1 score (raw data): %.2f%\n', F1_score_test_raw);
% 
% raw_data_time_taken = toc;
% fprintf('Time taken for knn on raw data: %f\n', raw_data_time_taken);
% 
% 
% function [accuracy, precision, recall, F1_score] = calculate_metrics(true_labels, predicted_labels)
%     % Calculate accuracy
%     accuracy = sum(predicted_labels == true_labels) / numel(true_labels);
% 
%     % Calculate confusion matrix
%     C = confusionmat(true_labels, predicted_labels);
% 
%     % Calculate precision and recall for each class
%     precision = zeros(1, size(C, 1));
%     recall = zeros(1, size(C, 1));
%     for i = 1:size(C, 1)
%         true_positive = C(i, i);
%         false_positive = sum(C(:, i)) - true_positive;
%         false_negative = sum(C(i, :)) - true_positive;
%         precision(i) = true_positive / (true_positive + false_positive);
%         recall(i) = true_positive / (true_positive + false_negative);
%     end
% 
%     % Calculate F1 Score
%     F1_score = 2 * (precision .* recall) ./ (precision + recall);
% end
