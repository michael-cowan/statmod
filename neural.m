clear all;
data = csvread('Data\MLB_batting_statsML.csv', 2, 7);
x = data(:, [1, 20]);
y = data(:, 4);