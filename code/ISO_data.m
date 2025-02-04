% Compute ISOmap
tic;
[iso_transformed, mapping_iso] = compute_mapping(z_scores_images, 'Isomap');
lle_time = toc;
fprintf('Time takenfor ISOmap: %.2f seconds\n', lle_time);