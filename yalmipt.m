% 清除变量
clear; clc; close all;
yalmip('clear');

% 参数定义
N = 40; % 总数据点数
N_train = 30; % 训练集数据点数
M = 1; % 铰链函数的数量
n_a = 1; % 输出滞后阶数
n_b = 1; % 输入滞后阶数
s = ones(M, 1); % 权重系数
M_theta = 10 * ones(M, 1); % z_{it} 的上界
theta_min = -5 * ones(n_a + n_b + 1, M); % theta_i 的下界
theta_max = 5 * ones(n_a + n_b + 1, M); % theta_i 的上界
D_bar = 5 * ones(M, 1); % 判别阈值
H_bar = randn(M, n_a + n_b); % 判别矩阵
m0 = 0.1 * ones(M, 1); % 小值

% 初始化输入和输出序列
y = zeros(N, 1); % 观测值 (输出)
u = randn(N, 1); % 随机生成输入值 (外生输入)

% 根据模型生成数据
for t = 2:N
    y(t) = 0.8 * y(t-1) + 0.4 * u(t-1) - 0.1 + max(-0.3 * y(t-1) + 0.6 * u(t-1) + 0.3, 0);
end

% for t = 2:N
%     y(t) = 0.83 * y(t-1) + 0.34 * u(t-1) - 0.2 + max(-0.34 * y(t-1) + 0.62 * u(t-1) + 0.4, 0);
% end

% for t = 2:N
%     y(t) = 0.82 * y(t-1) + 0.72 * u(t-1);
% end

% 将前 N_train 数据用于识别模型，其余数据用于预测
y_train = y(1:N_train);
u_train = u(1:N_train);
y_test = y(N_train+1:end);
u_test = u(N_train+1:end);

% 构建 phi 矩阵，直接包含常数项
phi = ones(n_a + n_b + 1, N);
for t = 2:N
    phi(2:end, t) = [y(t-1); u(t-1)];
end

% 构建训练集的 phi 矩阵，直接包含常数项
phi_train = ones(n_a + n_b + 1, N_train); % 初始化 phi 矩阵
% phi_train(1,1,1) = 1;
% phi_train(2,1,1) = 0;
% phi_train(3,1,1) = 0;
for t = 2:N_train
    phi_train(2:end, t) = [y_train(t-1); u_train(t-1)];
end

% 初始化 YALMIP 优化变量
varepsilon = sdpvar(N_train, 1); % 误差项 ε_t
theta_0 = sdpvar(n_a + n_b + 1, 1); % 基础回归系数 θ_0
theta = sdpvar(n_a + n_b + 1, M); % 分段线性函数的系数 θ_i
z = sdpvar(M, N_train, 'full'); % 辅助变量 z_{it}
delta = binvar(M, N_train, 'full'); % 二元变量 δ_{it}

% 目标函数
objective = sum(varepsilon);

% 初始化约束条件集合
constraints = [];

% 添加误差项的约束条件
for t = 1:N_train
    phi_t = phi_train(:, t); % 获取当前 phi_t
    constraints = [constraints, ...
        varepsilon(t) >= y_train(t) - phi_t' * theta_0 - sum(s .* z(:, t))];
    constraints = [constraints, ...
        varepsilon(t) >= phi_t' * theta_0 + sum(s .* z(:, t)) - y_train(t)];
end

% 添加变量的上下界及二元变量相关约束
for i = 1:M
    constraints = [constraints, theta_min(:, i) <= theta(:, i) <= theta_max(:, i)];
    for t = 2:N_train
        phi_t = phi_train(:, t);
        constraints = [constraints, ...
            z(i, t) >= 0, ...
            z(i, t) <= M_theta(i) * delta(i, t), ...
            phi_t' * theta(:, i) <= z(i, t), ...
            (1 - delta(i, t)) * m0(i) + z(i, t) <= phi_t' * theta(:, i)];
    end
end

% 设置求解器选项为 SCIP
options = sdpsettings('solver', 'scip', 'verbose', 1);

% 求解 MILP 问题
sol = optimize(constraints, objective, options);

% 检查求解状态
if sol.problem == 0
    disp('Optimal solution found:');
    optimal_varepsilon = value(varepsilon) % 最优误差项 ε_t
    optimal_theta_0 = value(theta_0)       % 最优 θ_0
    optimal_theta = value(theta)           % 最优 θ
else
    disp('Failed to find optimal solution.');
    disp(sol.info);
    return;
end

% 计算训练集的预测值
predicted_y_train = zeros(N_train, 1);
for t = 2:N_train
    phi_t = phi_train(:, t);
    predicted_y_train(t) = phi_t' * optimal_theta_0 + sum(s .* value(z(:, t)));
end

% 构建测试集的 phi 矩阵，直接包含常数项
phi_test = ones(n_a + n_b + 1, N - N_train);
for t = 2:(N - N_train)
    phi_test(2:end, t) = [y_test(t-1); u_test(t-1)];
end

% 计算测试集的预测值
predicted_y_test = zeros(N - N_train, 1);
predicted_y_test(1) = phi(:, N_train + 1)' * optimal_theta_0 + ...
sum(s .* max(phi(:, N_train + 1)' * optimal_theta, 0));
for t = 2:(N - N_train)
    phi_t_test = phi_test(:, t);
    predicted_y_test(t) = phi_t_test' * optimal_theta_0 + sum(s .* max(phi_t_test' * optimal_theta, 0));
end

% 绘制结果
figure;

% 1. 绘制训练集误差项 ε_t
subplot(3, 1, 1);
plot(1:N_train, optimal_varepsilon, '-o');
xlabel('Time step t');
ylabel('\epsilon_t');
title('Training Error term \epsilon_t');
grid on;

% 2. 对比训练集的预测值与实际观测值 y_t
subplot(3, 1, 2);
plot(1:N_train, y_train, 'b', 'DisplayName', 'Observed y (Train)');
hold on;
plot(1:N_train, predicted_y_train, 'r--', 'DisplayName', 'Predicted y (Train)');
xlabel('Time step t');
ylabel('y');
title('Observed vs Predicted Output (Training Set)');
legend('Location', 'best');
grid on;

% 3. 对比测试集的预测值与实际观测值 y_t
subplot(3, 1, 3);
plot(1:(N - N_train), y_test, 'b', 'DisplayName', 'Observed y (Test)');
hold on;
plot(1:(N - N_train), predicted_y_test, 'r--', 'DisplayName', 'Predicted y (Test)');
xlabel('Time step t');
ylabel('y');
title('Observed vs Predicted Output (Test Set)');
legend('Location', 'best');
grid on;

sgtitle('Visualization of Training and Test Results');
