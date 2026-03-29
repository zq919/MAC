addpath(genpath('MAC_fun'));
clc; clear;
try
    [sbCellType, sbxsC, sbysC, cellType, xsC, ysC] = ex_get_domain_case2_three_solid_circles();
    iPresBdry = 1;
    out = solve_StokesFlow(sbCellType, sbxsC, sbysC, iPresBdry);
catch ME
    fprintf('❌ 计算失败！错误信息:\n');
    fprintf('   错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
    fprintf('   错误信息: %s\n', ME.message);
end
%-------------------------------------------
rmpath(genpath('MAC_fun'))
%------------------------------------

function out = solve_StokesFlow(cellType, xsC, ysC, iPresBdry)
%边界也求
%SOLVE_STOKESFLOW  Mixed formulation Stokes on an embedded domain (MAC grid)
%   div(v)=0
%   mu*lap(v) - grad(p) + f = 0
%
% Inputs:
%   cellType  : size Nx-by-Ny, 0=fluid, <0=solid, >0=inlet/outlet id
%   xsC, ysC  : cell-center coordinates (vectors), uniform
%   iPresBdry : which inlet/outlet id uses p Dirichlet (as in Julia)
%
% Requires your framework functions/classes:
%   - AVariable_from_function(fun, {X,Y})  (or equivalent constructor)
%   - fd_restrict_unknown_inplace(avar, enabledMask)
%   - fd_register_unknown_multi_inplace([u v w ...])   % object array input
%   - fd_restrict_residual_inplace(residualAvar, enabledMask)
%   - fd_solve([eq1 eq2 eq3 ...])                      % object array input
%   - fd_get_solution(avar, raw_soln)

    paraMu  = 1.0;
    paraSrcX = 0.0; 
    paraSrcY = 0.0;

    % Dirichlet pressure boundary defined by cellType id:
    presBdry = @(iC,jC) 1.0 * (cellType(iC,jC) == iPresBdry);

    % ---- plots: cellType ----
    figure('Name','domain cellType');
    contourf(xsC, ysC, cellType.'); axis equal tight; colorbar;
    title('domain cellType');

    % ---- mesh steps (uniform) ----
    hx = xsC(2) - xsC(1);
    hy = ysC(2) - ysC(1);

    % node-line coordinates (MAC edges)
    xsNvec = [xsC(1) - hx/2, xsC + hx/2];   % length Nx+1
    ysNvec = [ysC(1) - hy/2, ysC + hy/2];   % length Ny+1

    % grids
    [XsC, YsC] = ndgrid(xsC,   ysC);    % Nx-by-Ny  (cell centers)
    [XsX, YsX] = ndgrid(xsNvec, ysC);   % (Nx+1)-by-Ny (x-edge centers)
    [XsY, YsY] = ndgrid(xsC,   ysNvec); % Nx-by-(Ny+1) (y-edge centers)
    [XsN, YsN] = ndgrid(xsNvec, ysNvec);% (Nx+1)-by-(Ny+1) (nodes)
    [nx, ny] = size(XsC);

    % ---- unknowns: pressure p ----
    p = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {XsC, YsC});

    enabled = (cellType == 0);  % fluid cells are unknowns for p
    assert(all(all(~enabled([1 end],:))) && all(all(~enabled(:,[1 end]))), ...
        'Require: physical boundary does NOT overlap box boundary (distance 1).');

    fd_restrict_unknown_inplace(p, enabled);

    % ---- unknowns: velx on x-edges (includes ghost edges, but masked out) ----
    velx = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {XsX, YsX});

    isEdgX0 = false(size(XsX));
    isEdgX1 = false(size(XsX));
    isEdgX  = false(size(XsX));

    % fill interior rows 2:end-1 from cell masks
    % 最后一列排除，因为边界在中心右边
    isEdgX0(2:end-1,:) = enabled(2:end,:)   & ~enabled(1:end-1,:); % west-facing boundary edges
    isEdgX1(2:end-1,:) = enabled(1:end-1,:) & ~enabled(2:end,:);   % east-facing boundary edges
    isEdgX (2:end-1,:) = enabled(2:end,:)   &  enabled(1:end-1,:); % interior edges

    enabledEgX = isEdgX | isEdgX0 | isEdgX1;
    fd_restrict_unknown_inplace(velx, enabledEgX);

    % ---- unknowns: vely on y-edges ----
    vely = AVariable_from_function(@(X,Y) 0.*X + 0.*Y, {XsY, YsY});

    isEdgY0 = false(size(XsY));
    isEdgY1 = false(size(XsY));
    isEdgY  = false(size(XsY));

    isEdgY0(:,2:end-1) = enabled(:,2:end)   & ~enabled(:,1:end-1); % south-facing boundary edges
    isEdgY1(:,2:end-1) = enabled(:,1:end-1) & ~enabled(:,2:end);   % north-facing boundary edges
    isEdgY (:,2:end-1) = enabled(:,2:end)   &  enabled(:,1:end-1); % interior edges

    enabledEgY = isEdgY | isEdgY0 | isEdgY1;
    fd_restrict_unknown_inplace(vely, enabledEgY);

    % ---- register unknowns into one global DOF vector ----
    fd_register_unknown_multi_inplace([p, velx, vely]);

    % ============================================================
    % Step 1: mass conservation  div(v)=0  (defined on cell centers)
    % ============================================================
    dvxdx = (velx(2:nx+1, :) - velx(1:nx, :)) / hx;   % Nx-by-Ny
    dvydy = (vely(:, 2:ny+1) - vely(:, 1:ny)) / hy;   % Nx-by-Ny
    eqMass = dvxdx + dvydy;

    % ============================================================
    % Step 2: momentum equations mu*lap(v) - grad(p) + f = 0
    % ============================================================

    % 2a) enforce dvxdx=0 and dvydy=0 on non-enabled cells (in/out + solid + ghost)
    Nx = numel(xsC); Ny = numel(ysC);
    for i = 1:Nx
        for j = 1:Ny
            if ~enabled(i,j)
                dvxdx(i,j) = 0.0 * velx(i,j);
                dvydy(i,j) = 0.0 * vely(i,j);
            end
        end
    end

    % 2b) dvxdy and dvydx with boundary treatment on nodes
    dvxdy = (velx(:, 2:ny) - velx(:, 1:ny-1)) / hy;  % (Nx+1)-by-(Ny-1)
    dvydx = (vely(2:nx, :) - vely(1:nx-1, :)) / hx;  % (Nx-1)-by-(Ny+1)

    for i = 2:(numel(xsNvec)-1)     % i = 2..Nx
        for j = 2:(numel(ysNvec)-1) % j = 2..Ny

            % --- correct dvxdy at (i, j-1) using cellType around node (i,j)
            if cellType(i-1,j-1) >= 0 && cellType(i,  j-1) >= 0 && ...
               (cellType(i-1,j) < 0 || cellType(i,  j) < 0)
                dvxdy(i, j-1) = (0.0 - velx(i, j-1)) / (hy/2.0);

            elseif cellType(i-1,j) >= 0 && cellType(i,  j) >= 0 && ...
                   (cellType(i-1,j-1) < 0 || cellType(i,  j-1) < 0)
                dvxdy(i, j-1) = (velx(i, j) - 0.0) / (hy/2.0);

            elseif cellType(i-1,j) > 0 || cellType(i, j) > 0 || ...
                   cellType(i-1,j-1) > 0 || cellType(i, j-1) > 0
                dvxdy(i, j-1) = 0.0 * velx(i, j);
            end

            % --- correct dvydx at (i-1, j)
            if cellType(i-1,j-1) >= 0 && cellType(i-1,j) >= 0 && ...
               (cellType(i-1,j) < 0 || cellType(i,  j) < 0)
                dvydx(i-1, j) = (0.0 - vely(i-1, j)) / (hx/2.0);

            elseif cellType(i,  j-1) >= 0 && cellType(i,  j) >= 0 && ...
                   (cellType(i-1,j-1) < 0 || cellType(i-1,j) < 0)
                dvydx(i-1, j) = (vely(i, j) - 0.0) / (hx/2.0);

            elseif cellType(i, j-1) > 0 || cellType(i, j) > 0 || ...
                   cellType(i-1,j-1) > 0 || cellType(i-1,j) > 0
                dvydx(i-1, j) = 0.0 * vely(i, j);
            end
        end
    end

    % 2c) dpdx and dpdy with bdry treatment (in/out only)
    dpdx = (p(2:nx, :) - p(1:nx-1, :)) / hx;  % (Nx-1)-by-Ny
    for i = 2:(numel(xsNvec)-1)  % 2..Nx
        for j = 1:Ny
            if isEdgX0(i,j)
                dpdx(i-1, j) = (p(i, j) - presBdry(i-1, j)) / (hx/2.0);
            elseif isEdgX1(i,j)
                dpdx(i-1, j) = (presBdry(i, j) - p(i-1, j)) / (hx/2.0);
            end
        end
    end

    dpdy = (p(:, 2:ny) - p(:, 1:ny-1)) / hy;  % Nx-by-(Ny-1)
    for i = 1:Nx
        for j = 2:(numel(ysNvec)-1)  % 2..Ny
            if isEdgY0(i,j)
                dpdy(i, j-1) = (p(i, j) - presBdry(i, j-1)) / (hy/2.0);
            elseif isEdgY1(i,j)
                dpdy(i, j-1) = (presBdry(i, j) - p(i, j-1)) / (hy/2.0);
            end
        end
    end

    % 2d) laplacian terms (second derivatives)
    dvxdxx = (dvxdx(2:nx, :) - dvxdx(1:nx-1, :)) / hx;        % (Nx-1)-by-Ny
    dvxdyy = (dvxdy(:, 2:ny-1) - dvxdy(:, 1:ny-2)) / hy;        % (Nx+1)-by-(Ny-2)
    dvydxx = (dvydx(2:nx-1, :) - dvydx(1:nx-2, :)) / hx;        % (Nx-2)-by-(Ny+1)
    dvydyy = (dvydy(:, 2:ny) - dvydy(:, 1:ny-1)) / hy;        % Nx-by-(Ny-1)

    eqMomtX = paraMu * (dvxdxx(:, 2:ny-1) + dvxdyy(2:nx, :)) ...
              - dpdx(:, 2:ny-1) + paraSrcX;                    % (Nx-1)-by-(Ny-2) 只计算内部需要求的，最外层幽灵点补求

    eqMomtY = paraMu * (dvydxx(:, 2:ny) + dvydyy(2:nx-1, :)) ...
              - dpdy(2:nx-1, :) + paraSrcY;                    % (Nx-2)-by-(Ny-1)

    % 2e) no-slip on solid boundaries: enforce vel = 0 by overwriting momentum eq.
    for i = 2:(numel(xsNvec)-1)      % 2..Nx
        for j = 2:(numel(ysC)-1)     % 2..Ny-1
            if isEdgX0(i,j)
                if cellType(i-1,j) < 0
                    eqMomtX(i-1, j-1) = velx(i,j) - 0.0;
                end
            end
            if isEdgX1(i,j)
                if cellType(i, j) < 0
                    eqMomtX(i-1, j-1) = velx(i,j) - 0.0;
                end
            end
        end
    end

    for i = 2:(numel(xsC)-1)         % 2..Nx-1
        for j = 2:(numel(ysNvec)-1)  % 2..Ny
            if isEdgY0(i,j)
                if cellType(i, j-1) < 0
                    eqMomtY(i-1, j-1) = vely(i,j) - 0.0;
                end
            end
            if isEdgY1(i,j)
                if cellType(i, j) < 0
                    eqMomtY(i-1, j-1) = vely(i,j) - 0.0;
                end
            end
        end
    end

    % ---- restrict residual equations to active locations ----
    fd_restrict_residual_inplace(eqMass, enabled);
    fd_restrict_residual_inplace(eqMomtX, enabledEgX(2:end-1, 2:end-1));
    fd_restrict_residual_inplace(eqMomtY, enabledEgY(2:end-1, 2:end-1));

    % ---- solve ----
    raw_soln = fd_solve([eqMass, eqMomtX, eqMomtY]);

    p_soln  = fd_get_solution(p, raw_soln);
    vx_soln = fd_get_solution(velx, raw_soln);
    vy_soln = fd_get_solution(vely, raw_soln);

    % ============================================================
    % Flow rates on inlets/outlets
    % ============================================================
    numInOutLets = max(cellType(:));
    flowOutRates = zeros(numInOutLets, 1);

    for i = 2:(numel(xsNvec)-1)
        for j = 2:(numel(ysC)-1)
            if isEdgX0(i,j) && cellType(i-1,j) > 0
                flowOutRates(cellType(i-1,j)) = flowOutRates(cellType(i-1,j)) + (-vx_soln(i,j))*hy;
            end
            if isEdgX1(i,j) && cellType(i,j) > 0
                flowOutRates(cellType(i,j)) = flowOutRates(cellType(i,j)) + ( vx_soln(i,j))*hy;
            end
        end
    end

    for i = 2:(numel(xsC)-1)
        for j = 2:(numel(ysNvec)-1)
            if isEdgY0(i,j) && cellType(i,j-1) > 0
                flowOutRates(cellType(i,j-1)) = flowOutRates(cellType(i,j-1)) + (-vy_soln(i,j))*hx;
            end
            if isEdgY1(i,j) && cellType(i,j) > 0
                flowOutRates(cellType(i,j)) = flowOutRates(cellType(i,j)) + ( vy_soln(i,j))*hx;
            end
        end
    end

    % ---- plot pressure ----
    fig_pres = figure('Name','pressure solution');
    contourf(xsC, ysC, p_soln.'); axis equal tight; colorbar;
    title('pressure solution');

    % ---- plot velocity (node-centered by averaging) ----
    vx = 0.5*(vx_soln(2:end-1, 1:end-1) + vx_soln(2:end-1, 2:end)); % (Nx-1)-by-(Ny-1)
    vy = 0.5*(vy_soln(1:end-1, 2:end-1) + vy_soln(2:end, 2:end-1)); % (Nx-1)-by-(Ny-1)

    XsNd = XsN(2:end-1, 2:end-1);
    YsNd = YsN(2:end-1, 2:end-1);

    % downsample for quiver
    stride = 4;
    i_idx = 1:stride:size(XsNd,1);
    j_idx = 1:stride:size(XsNd,2);

    XsNd_sub = XsNd(i_idx, j_idx);
    YsNd_sub = YsNd(i_idx, j_idx);
    vx_sub   = vx(i_idx, j_idx);
    vy_sub   = vy(i_idx, j_idx);

    magnitudes = sqrt(vx_sub.^2 + vy_sub.^2);
    maxArrowLength = 0.2*0.2;
    mmax = max(magnitudes(:));
    if mmax == 0
        scale = 1.0;
    else
        scale = mmax / maxArrowLength;
    end

    fig_vel = figure('Name','velocity solution');
    quiver(XsNd_sub, YsNd_sub, vx_sub/scale, vy_sub/scale, 0, 'k');
    axis equal tight;
    title('velocity solution');

    % ---- pack outputs ----
    out = struct();
    out.flowOutRates = flowOutRates;
    out.raw_soln = raw_soln;

    out.cellType = cellType;
    out.enabled = enabled;
    out.enabledEgX = enabledEgX;
    out.enabledEgY = enabledEgY;

    out.p = p;
    out.velx = velx;
    out.vely = vely;

    out.eqMass  = eqMass;
    out.eqMomtX = eqMomtX;
    out.eqMomtY = eqMomtY;

    out.p_soln  = p_soln;
    out.vx_soln = vx_soln;
    out.vy_soln = vy_soln;

    out.fig_pres = fig_pres;
    out.fig_vel  = fig_vel;

    out.XsN = XsN;
    out.YsN = YsN;
end

function [sbCellType, sbXsC, sbYsC, cellType, xsC, ysC] = ex_get_domain_case2_three_solid_circles()
%EX_GET_DOMAIN_CASE2_THREE_SOLID_CIRCLES
% Domain: Omega0 = [a0,a1] x [b0,b1]
% Solid obstacles: union of 3 circles (marked as negative cellType)
% Partition of outer boundary types: by "right side" of directed lines between centers (marked 1/2/3)
% Fluid cells: cellType == 0
% Then extract a tight subdomain containing all fluid cells with a 1-cell padding.

    % ---- box domain ----
    a0 = 1.0; a1 = 2.0;
    b0 = 5.0; b1 = 6.0;

    % ---- three circle centers (rows) and radii ----
    cirs_ctr = [1.1 5.1;
                1.9 5.3;
                1.5 5.9];
    cirs_r   = [0.4, 0.3, 0.3];

    % ensure CCW arrangement of centers
    assert(det([ones(3,1), cirs_ctr]) > 0, ...
        'Circle centers must be counter-clockwise (detTriangle>0).');

    % ---- grid resolution ----
    nx = 320; ny = 320;   % same as your Julia defaults
%     nx = 640; ny = 640; % can be heavier

    % ---- cell centers ----
    xs  = linspace(a0, a1, nx+1);
    ys  = linspace(b0, b1, ny+1);
    xsC = 0.5*(xs(1:end-1) + xs(2:end));
    ysC = 0.5*(ys(1:end-1) + ys(2:end));

    % ndgrid gives Xs(i,j) with i along xsC, j along ysC (matches Julia comprehension)
    [Xs, Ys] = ndgrid(xsC, ysC);  % size nx-by-ny

    % ---- cellType initialization ----
    cellType = zeros(size(Xs), 'int32');

    % ---- helper: "right side of directed segment (A->B)" test, vectorized ----
    % right side <=> cross(B-A, P-A) < 0  (2D)
    isInRight = @(X,Y, i_cir, j_cir) ...
        ((cirs_ctr(j_cir,1) - cirs_ctr(i_cir,1)) .* (Y - cirs_ctr(i_cir,2)) ...
       - (cirs_ctr(j_cir,2) - cirs_ctr(i_cir,2)) .* (X - cirs_ctr(i_cir,1))) < 0;

    % ---- helper: inside circle test, vectorized ----
    isInCir = @(X,Y, i_cir) ...
        ((X - cirs_ctr(i_cir,1)).^2 + (Y - cirs_ctr(i_cir,2)).^2) < (cirs_r(i_cir)^2);

    % ---- set boundary region IDs by half-plane tests (order matters, later can overwrite) ----
    cellType(isInRight(Xs, Ys, 1, 2)) = 1;
    cellType(isInRight(Xs, Ys, 2, 3)) = 2;
    cellType(isInRight(Xs, Ys, 3, 1)) = 3;

    % ---- mark solid circles as negative IDs (overwrite any previous 1/2/3) ----
    for i_cir = 1:size(cirs_ctr,1)
        maskCir = isInCir(Xs, Ys, i_cir);
        cellType(maskCir) = -i_cir;
    end

    % ---- find tight subdomain around fluid cells (cellType == 0), padded by 1 cell ----
    [I0, J0] = find(cellType == 0);
    assert(~isempty(I0), 'No fluid cells (cellType==0) found.');

    iMin = min(I0) - 1;  iMax = max(I0) + 1;
    jMin = min(J0) - 1;  jMax = max(J0) + 1;

    % clamp & assert inside bounds (Julia had assert; here we clamp safely then assert)
    iMin = max(iMin, 1); iMax = min(iMax, size(cellType,1));
    jMin = max(jMin, 1); jMax = min(jMax, size(cellType,2));

    assert(iMin >= 1 && iMax <= size(cellType,1) && jMin >= 1 && jMax <= size(cellType,2), ...
        'Subdomain indices out of bounds.');

    sbCellType = cellType(iMin:iMax, jMin:jMax);
    sbXsC = xsC(iMin:iMax);
    sbYsC = ysC(jMin:jMax);

    % ---- plots (match Julia: contourf(xsC, ysC, cellType') ) ----
    fig1 = figure('Name','domain cellType');
    contourf(xsC, ysC, double(cellType).');  % transpose for (x,y) convention
    axis equal tight; colorbar;
    title('domain cellType');

    fig2 = figure('Name','subdomain cellType');
    contourf(sbXsC, sbYsC, double(sbCellType).');
    axis equal tight; colorbar;
    title('subdomain cellType');

    % ---- optional save (similar to save_plot_with_dir) ----
    save_plot_with_dir(fig1, fullfile('ziquyutiqu','test_2.png'), 'results');
    save_plot_with_dir(fig2, fullfile('ziquyutiqu','test_3.png'), 'results');
end

% ============================================================
% Helper: oriented area (2D cross product) for triangle [P1;P2;P3]
% If >0 => counter-clockwise
% ============================================================
function detv = get_detTriangle(P)
% P: 3x2 matrix, rows are points [x y]
    assert(all(size(P) == [3 2]), 'get_detTriangle expects a 3x2 matrix.');
    x1 = P(1,1); y1 = P(1,2);
    x2 = P(2,1); y2 = P(2,2);
    x3 = P(3,1); y3 = P(3,2);
    detv = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1);
end

% ============================================================
% Helper: save figure to folder/relativepath (mkdir automatically)
% ============================================================
function save_plot_with_dir(figH, relPath, folder)
    if nargin < 3 || isempty(folder)
        folder = '.';
    end
    outPath = fullfile(folder, relPath);
    outDir = fileparts(outPath);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    % For R2020a+ recommended:
    try
        exportgraphics(figH, outPath, 'Resolution', 200);
    catch
        saveas(figH, outPath);
    end
end
