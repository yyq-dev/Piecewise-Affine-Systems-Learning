%% 使用混合整数规划（MILP）拟合数据示例
% 本脚本利用 YALMIP 和 SCIP 求解器，对前 100 个数据点进行参数估计

clear; clc; close all;

%% 1. 数据生成：单位阶跃响应 y(t)=1-exp(-t)
T = 5;                % 取时间范围 [0, T]
Ndata = 100;          % 数据点个数
t = linspace(0, T, Ndata)';  % 列向量
y_true = 1 - exp(-t);        % 单位阶跃响应

%% 2. 构造分段线性模型
% 模型形式： f(t)= c + sum_{j=1}^{Nbp} d_j * (t - tau_j)_+
% 其中 (t - tau_j)_+ = max(0, t-tau_j)

Nbp = 10;                      % 候选断点数量
tau = linspace(0, T, Nbp)';      % 断点位置（列向量）

% 预先计算 hinge 函数值
H = zeros(Ndata, Nbp);
for j = 1:Nbp
    H(:,j) = max(0, t - tau(j));
end

%% 3. 建立 MILP 模型（使用 YALMIP）
% 定义决策变量
c = sdpvar(1,1);         % 截距
d = sdpvar(Nbp,1);       % 各 hinge 函数的系数
z = binvar(Nbp,1);       % 二进制变量：若 z(j)=0 则 d(j)=0

% 定义每个数据点的拟合值： f_i = c + sum_j d(j)*H(i,j)
f = c + H*d;

% 为了采用 L1 范数拟合，引入辅助变量 e(i) 表示 |f(i)-y_true(i)|
e = sdpvar(Ndata,1);

Constraints = [];
% 绝对值约束
for i = 1:Ndata
    Constraints = [Constraints, e(i) >= f(i) - y_true(i), ...
                              e(i) >= -(f(i) - y_true(i))];
end

% 采用大 M 技巧将 d 与二进制变量 z 关联，确保当 z(j)=0 时 d(j)=0
M = 100;   % 合适的足够大常数，可根据实际情况调整
for j = 1:Nbp
    Constraints = [Constraints, d(j) <= M*z(j), d(j) >= -M*z(j)];
end

% 限制最多使用 K 个断点（即最多允许 K 个非零的 d(j)）
K = 5;
Constraints = [Constraints, sum(z) <= K];

% 目标函数：最小化所有数据点的绝对误差之和
Objective = sum(e);

%% 4. 求解 MILP 问题
ops = sdpsettings('solver','scip','verbose',1);
sol = optimize(Constraints, Objective, ops);

if sol.problem == 0
    disp('求解成功');
else
    disp('求解遇到问题');
    sol.info
end

% 取出拟合结果
c_fit = value(c);
d_fit = value(d);
fitted = value(f);

%% 5. 绘图比较原曲线与拟合曲线
figure;
plot(t, y_true, 'b-', 'LineWidth', 2); hold on;
plot(t, fitted, 'r--', 'LineWidth', 2);
legend('真实曲线: 1-exp(-t)', 'MILP 拟合曲线', 'Location', 'southeast');
xlabel('时间 t (s)');
ylabel('响应 y');
title('拟合单位阶跃响应曲线');
grid on;
