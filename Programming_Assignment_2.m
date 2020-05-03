%
% References
% http://www.miviclab.org/docs/MTSC_852_Lab_Sessions.pdf
% https://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
% https://www.mathworks.com/help/stats/distribution-plots.html
% https://www.mathworks.com/help/matlab/ref/cov.html#buo3tdl-1
% https://www.mathworks.com/help/matlab/ref/eig.html
% https://www.mathworks.com/help/stats/normplot.html
% https://www.mathworks.com/matlabcentral/answers/67226-how-to-do-a-classification-using-matlab
% https://www.mathworks.com/help/matlab/ref/reshape.html
% https://www.mathworks.com/help/matlab/creating_plots/add-title-axis-labels-and-legend-to-graph.html
% https://www.mathworks.com/matlabcentral/answers/108246-change-color-of-points
% https://www.csd.uwo.ca/~olga/Courses/CS434a_541a/Lecture8.pdf
% https://www.mathworks.com/help/matlab/ref/mean.html
% https://sthalles.github.io/fisher-linear-discriminant/
% https://www.mathworks.com/help/matlab/ref/axis.html
% https://www.mathworks.com/help/matlab/ref/repmat.html
% https://stackoverflow.com/questions/16146212/how-to-plot-a-hyper-plane-in-3d-for-the-svm-results/19969412#19969412
% https://www.mathworks.com/help/matlab/ref/ellipsoid.html
% https://www.mathworks.com/matlabcentral/answers/141965-how-do-i-connect-points-in-a-scatter-plot-with-a-line
% https://www.mathworks.com/help/matlab/ref/yline.html
% https://www.mathworks.com/matlabcentral/answers/353155-wblfit-error-x-must-be-a-vector-containing-positive-values-but-vectors-contain-only-positive-non

clc;
points = load('points.dat');
total_class = 3; % total class in data set
class1 = 2; % class number
class2 = 3; % class number
[w, X1, X2, Y1, Y2] = FLD(points, total_class, class1 , class2);

fprintf("Question B : Optimal Line Direction w \n");
disp(w);

plot3(points(:, 4)', points(:, 5)', points(:, 6)','bo','linewidth', 1);
hold on;
plot3(points(:, 7)', points(:, 8)', points(:, 9)','go','linewidth', 1);
view(2);
grid on;
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');
title('Question C : Scattered Points', 'fontsize', 15);

figure, plot3(Y1,Y1,Y1, '-o');
hold on;
plot3(Y2,Y2,Y2, '-o');
view(2);
legend({'w2','w3'},'Location','northeast');
grid on;
xlabel('x-axis')
ylabel('y-axis')
title('Question C : Projection of Scattered points onto line', 'fontsize', 15);

A = zeros(1,20);
A(:, 1 : 10) = sort(Y1);
A(:, 11 : 20) = sort(Y2);
[mu_2,sigma_2] = normfit(sort(Y1));
pdf_2= normpdf(A, mu_2,sigma_2);
[mu_3,sigma_3] = normfit(sort(Y2));
pdf_3 = normpdf(A, mu_3,sigma_3);
figure, plot(sort(Y1),pdf_2(1:10), '-o');
hold on;
plot(sort(Y2),pdf_3(11:20), '-o');
Boundary_Line = xline(0.014986,'-',"Boundary");
Boundary_Line.Color = 'r';
legend({'w2','w3'},'Location','northeast');
title('Question D : Gaussian Distribution', 'fontsize', 18);
xlabel('x-axis');
ylabel('y-axis');
grid on;

figure, plot(Y1, ones(10, 1),'bo');
hold on;
plot(Y2, ones(10, 1), 'go', 'linewidth', 2);
Boundary_Line = xline(0.014986,'-',"Boundary");
Boundary_Line.Color = 'r';
legend({'w2','w3'},'Location','northeast');
xlabel('x-axis');
ylabel('y-axis');
grid on;
title('Question D : Decision Boundary', 'fontsize', 18);
hold;
fprintf("Question D :Decision  Boundary  0.014986\n");

% training error calculation.
d = pdf_2 < pdf_3;
d = d + 2;
fprintf('Question E : Training Error set w2\n');
disp(d(:, 1 : 10));
fprintf('Question E : Training Error set w3\n');
disp(d(:, 11 : 20));
classification_rate = sum( d == [2* ones(1,10),3* ones(1,10)]) / numel([2* ones(1,10),3* ones(1,10)]);
fprintf('Question E : Error rate = %f (in percentage)\n',100 * (1 - classification_rate));

w1 = [1, 2 , -1.5]';
New_Y1 = w1' * X1;
New_Y2 = w1' * X2;

New_A = zeros(1,20);
New_A(:, 1 : 10) = sort(New_Y1);
New_A(:, 11 : 20) = sort(New_Y2);
[New_mu_2,New_sigma_2] = normfit(sort(New_Y1));
New_pdf_2= normpdf(New_A, New_mu_2,New_sigma_2);
[New_mu_3,New_sigma_3] = normfit(sort(New_Y2));
New_pdf_3 = normpdf(New_A, New_mu_3,New_sigma_3);
figure, %plot(sort(New_Y1),New_pdf_2(1:10), '-');
s = rng;
r = sort(normrnd(New_mu_2,New_sigma_2,[1,100]));
rr = normpdf(r, New_mu_2, New_sigma_2);
plot(sort(r),rr);

hold on;
%plot(sort(New_Y2),New_pdf_3(11:20), '-');
s = rng;
r = sort(normrnd(New_mu_3,New_sigma_3,[1,100]));
rr = normpdf(r, New_mu_3, New_sigma_3);
plot(sort(r),rr);
Boundary_Line = xline(-0.1501,'-');
Boundary_Line.Color = 'r';
Boundary_Line = xline(1.621,'-');
Boundary_Line.Color = 'r';
legend({'w2','w3'},'Location','northeast');
title('Question F (Part D) : Gaussian Distribution', 'fontsize', 18);
xlabel('x-axis');
ylabel('y-axis');
grid on;

figure, plot(New_Y1, ones(10, 1),'bo', 'linewidth', 1);
hold on;
plot(New_Y2, ones(10, 1), 'go', 'linewidth', 2);
New_x = 0.081564;
Boundary_Line = xline(0.081564,'-',"Boundary");
Boundary_Line.Color = 'r';
legend({'w2','w3','Boundary Line'},'Location','northeast');
xlabel('x-axis');
ylabel('y-axis');
grid on;
title('Question F (Part D)', 'fontsize', 18);
fprintf("Question F (Part D) Decision Boundary %f\n",New_x);

d = New_pdf_2 < New_pdf_3;
d = d + 2;
fprintf('Question F (Part D) : Non Optimal Subspace set w2\n');
disp(d(:, 1 : 10));
fprintf('Question F (Part D) : Non Optimal Subspace set w3\n');
disp(d(:, 11 : 20));
classification_rate = sum( d == [2* ones(1,10),3* ones(1,10)]) / numel([2* ones(1,10),3* ones(1,10)]);
fprintf('Question F (Part E): Non Optimal Subspace Error rate =  %f (in percentage) \n',100 * (1 - classification_rate));

function [w, X1, X2, Y1, Y2] = FLD(points, total_class, class1, class2)

[row, column] = size(points);
dimension = column / total_class;

fc1 = dimension *( class1 - 1) + 1;
lc1 = dimension * class1;

fc2 = dimension *( class2 - 1) + 1;
lc2 = dimension * class2;

X1 = points( : , fc1 : lc1)';
X2 = points( : , fc2 : lc2)';

mean_vector( : , 2 ) = mean(X1, 2);
M = repmat(mean_vector( : , class1 ), 1, row);
S{1} = (X1 - M) * (X1 - M)';

mean_vector(:, 3) = mean(X2, 2);
M = repmat(mean_vector( : , class2 ), 1, row);
S{2} = (X2 - M) * (X2 - M)';

Sw =S{1} + S{2};
w = inv(Sw) * ( mean_vector( : , class1 ) - mean_vector( : , class2 ) );

Y1 = w' * X1;
Y2 = w' * X2;

end
