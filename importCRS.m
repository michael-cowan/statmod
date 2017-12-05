function f = importCOS(p)
    f = csvread(['Data\COS_prev' num2str(p) '.csv'], 1, 0);