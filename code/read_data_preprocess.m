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