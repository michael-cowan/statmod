main = csvread('Data\MLB_batting_statsML.csv', 2, 6);
G_avg = main(:, [1, 20]).';
H = main(:, 4).';
% 
% crs = importCRS(9);
% x = crs(:, 1:end-1).';
% y = crs(:, end).';