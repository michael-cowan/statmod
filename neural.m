% main = csvread('Data\MLB_batting_statsML.csv', 2, 6);
% G_avg = main(:, [1, 20]).';
% H = main(:, 4).';

cos = importCOS(9);
x = cos(:, 1:end-1).';
y = cos(:, end).';