test_vals = categorical({'1','2','5','10','20','30','40','50','75','99'});
test_vals = reordercats(test_vals,{'1','2','5','10','20','30','40','50','75','99'});

beta_1_12_5 = [59.78, 76.82, 84.28, 82.04, 38.86, ...
    24.78, 24.42, 26.1, 18.3, 15.26];

exp_1_12_5 = [65.72, 76.88, 82.7, 82.48, 44.22, 27.44, 27.28, 23.84, 10.68, 16.38];
unif_1_12_5 = [62.36 75.64 83.72 85.32 46.86 26.7 33.08 23.44 15.42 11.16];
combined = [beta_1_12_5; exp_1_12_5; unif_1_12_5];
figure, bar( test_vals, combined), xlabel("Testing Rate (%)"), 
ylabel("Accuracy (%)"), ylim([0,100]); legend("Beta", "Exp", "Unif");
title("Training Rates Varying from 1 - 12.5%");

beta_all_training_rates = [59.78, 76.82, 84.28, 82.04, 38.86, 24.78, 24.42, 26.1, 18.3, 15.26;...
15.38 21.26 31.44 72.16 87.06 73.92 49.26 45.98 37.08 27.38;...
57.88 75.1 83.66 85.64 83.84 60.18 36.32 34.42 31.2 28.94;...
11.72 15.5 26.2 37.68 73.1 88.28 79.2 72.64 58.78 46.22;...
9.72 10 22.2 31 49.54 78.8 89.48 85.64 67.52 57.72;...
8.46 13.54 19.86 36.14 65.98 88.7 88.22 82.08 63.22 50.62;...
51.8 72.18 82.42 84.96 86.76 86.26 83.4 74.44 49.42 35.38];

exp_all_training_rates = ...
[65.72, 76.88, 82.7, 82.48, 44.22, 27.44, 27.28, 23.84, 10.68, 16.38; ...
16.7 15.32 33.26 68.96 87.98 78.48 58.62 48.34 43.7 35.22; ...
64.24 75.96 83.56 84.14 83.64 77.28 50.74 32.2 23.5 26.72; ...
7.68, 9.06 26.6 43.08 77.74 89.22 87.54 82.24 68.76 60.88; ...
11.5 14.86 19.24 37.32 49.7 79.74 89.08 89.48 78.82 65.54; ...
10.7 9.4 25.64 39.16 75.26 88.74 88.46 88.04 77.28 65.8; ...
64.02 75.22 82.96 85.24 85.9 85.6 84.22 84.74 69.8 45.24];

unif_all_training_rates = [62.36 75.64 83.72 85.32 46.86 26.7 33.08 23.44 15.42 11.16;
14.96 19.42 27 72.36 88.14 81.08 71.4 60.38 48.32 36.88;...
61.7 74.4 83.24 85.44 86.96 78.98 53.38 40.32 30.6 30.1;...
12.28 9.66 27.8 40.5 75.52 88.32 86.14 80.36 63.4 54.5;...
9.96 9.76 13.96 34.6 58.52 78.62 89.18 89.6 74.48 67.02;...
9 7.34 24.02 36.68 76.34 88.5 89.44 88.58 77.88 68.08;...
59.58 72.24 81.6 84.72 86.76 87.58 87.92 86.86 76.6 71.28];

tr_titles = ["1-12.5%", "12.5-25%", "1-25%","25-37.5", "37.5-50%", "25-50%", "1-50%"];

figure,
for j = 1:size(exp_all_training_rates, 1)
    subplot(2,4,j), bar(test_vals, exp_all_training_rates(j,:)),
    title(tr_titles(j)),xlabel("Testing Rate (%)"), 
    ylabel("Accuracy (%)"), ylim([0,100]);
end
sgtitle("Varying Training Rates Exponential Distribution")

figure,
for j = 1:size(beta_all_training_rates, 1)
    subplot(2,4,j), bar(test_vals, beta_all_training_rates(j,:)),
    title(tr_titles(j)),xlabel("Testing Rate (%)"), 
    ylabel("Accuracy (%)"), ylim([0,100]);
end
sgtitle("Varying Training Rates Beta Distribution")

figure,
for j = 1:size(unif_all_training_rates, 1)
    subplot(2,4,j), bar(test_vals, unif_all_training_rates(j,:)),
    title(tr_titles(j)),xlabel("Testing Rate (%)"), 
    ylabel("Accuracy (%)"), ylim([0,100]);
end
sgtitle("Varying Training Rates Uniform Distribution")

