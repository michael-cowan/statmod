clear all;
data = csvread('Data\MLB_batting_statsML.csv', 2, 6);
G_avg = data(:, [1, 20]).';
H = data(:, 4).';