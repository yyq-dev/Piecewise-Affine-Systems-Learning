clear;clc;close all;

% 设置随机种子
rng(42);

% 时间序列长度
n = 100;

% 生成目标时间序列 y
time = (1:n)';
y = 0.5 * time + sin(time) + randn(n, 1);

% 生成外生变量 X
X = 0.3 * time + randn(n, 1);

% y = randn(n, 1);
% X = randn(n, 1);

% 可视化数据
figure;
plot(y, 'DisplayName', 'Target (y)');
hold on;
plot(X, '--', 'DisplayName', 'Exogenous Variable (X)');
legend;
title('Time Series Data');
hold off;

% 将前80个数据点作为训练集，后20个数据点作为测试集
train_size = floor(0.8 * n);
y_train = y(1:train_size);
X_train = X(1:train_size);
y_test = y(train_size+1:end);
X_test = X(train_size+1:end);

% 自回归阶数
p = 1;

% 构建设计矩阵 Z
y_lag = y_train(1:end-1);    % 自回归项 y_{t-1}
X_lag = X_train(2:end);      % 外生变量 X_t

% 构建设计矩阵（每行包含 [y_{t-1}, X_t]）
Z = [y_lag, X_lag];

% 构建目标向量（去掉第一个数据点，因为没有 y_{t-1}）
y_target = y_train(2:end);

% 使用最小二乘法拟合 ARX 模型的系数
coefficients = Z \ y_target;

% 显示拟合得到的系数
fprintf('AR coefficient (alpha): %.4f\n', coefficients(1));
fprintf('Exogenous variable coefficient (beta): %.4f\n', coefficients(2));

% 使用测试集数据进行预测
y_pred = zeros(length(y_test), 1);

% 使用最后一个训练集的点作为初始自回归项
y_pred(1) = coefficients(1) * y_train(end) + coefficients(2) * X_test(1);

% 对测试集中的每一个点进行预测
for t = 2:length(y_test)
    y_pred(t) = coefficients(1) * y_pred(t-1) + coefficients(2) * X_test(t);
end

% 计算均方误差（MSE）
mse = mean((y_test - y_pred).^2);
fprintf('Mean Squared Error: %.4f\n', mse);

% 可视化预测结果
figure;
plot(y_test, 'DisplayName', 'Actual');
hold on;
plot(y_pred, '--', 'DisplayName', 'Predicted');
legend;
title('ARX Model Predictions vs Actual');
hold off;
